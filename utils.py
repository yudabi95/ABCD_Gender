import os
from tensorboardX import SummaryWriter
from opt_params import parser,args
from scipy.ndimage.interpolation import zoom
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from model import AlexNet3D_Dropout
import numpy as np
import pandas as pd
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, explained_variance_score, mean_squared_error, r2_score
import nibabel as nib
from scipy.ndimage.interpolation import zoom


from torch.optim.lr_scheduler import ReduceLROnPlateau


writer = SummaryWriter()

torch.cuda.set_device('cuda:1')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class SMRIDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(root_dir + csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 8])
        image = np.float32(nib.load(img_path).get_fdata())
        image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
        image = torch.from_numpy(image)
        y_label = self.annotations.iloc[index, 2]
        if y_label == 'F':
            y_label = 0
        else:
            y_label = 1

        if self.transform:
            image = self.transform(image)

        return image, y_label


def loadNet():

    # Load validated model
    net = initializeNet()
    model = torch.nn.DataParallel(net, device_ids = [1])
    net = 0
    net = load_net_weights2(model, '/home/users/ybi3/SMLvsDL/Alex_Pre-Trained_Adam_1.pt')

    return net

def loadNetWithPara():

    net = initializeNet()
    model = torch.nn.DataParallel(net, device_ids=[1])

    return model


def initializeNet():

    net = AlexNet3D_Dropout(num_classes=2)

    return net

def load_net_weights2(net, weights_filename):

    # Load trained model
    state_dict = torch.load(
        weights_filename,  map_location=device)
    state = net.state_dict()
    state.update(state_dict)
    net.load_state_dict(state_dict)

    return net

# Train Network
def train(dataloader, model,criterion,optimizer):


    train_loss = 0
    model.train()

    for batch_idx, (data,targets) in enumerate(dataloader):
        # Get data to cuda if possible
        data = data.to(device)
        targets = targets.to(device)

        # forward
        scores = model(data)
        loss = criterion(scores[0].squeeze(), targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        train_loss += loss.item()
        # gradient descent or adam step
        optimizer.step()


    return train_loss



# Check accuracy on training to see how good our model is
def test(dataloader, model):

    model.eval()

    y_pred = np.array([])
    y_true = np.array([])

    for batch_idx, (inputs, labels) in enumerate(dataloader):

        inputs = inputs.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(inputs)
        _, predicted = torch.max(outputs[0].data, 1)
        y_pred = np.concatenate((y_pred, predicted.cpu().numpy()))
        y_true = np.concatenate((y_true, labels.data.cpu().numpy()))

        print('processing the accuracy per epochs, this is the',batch_idx,'batch')

    return y_true, y_pred

def evalMetrics(dataloader, model):

    y_true, y_pred = test(dataloader, model)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    return acc, bal_acc

def loaddata():

    train_set = SMRIDataset(
        csv_file="tr_9194_rep_4.csv",
        root_dir="/home/users/ybi3/SMLvsDL/SampleSplits_Sex/",
    )

    val_set = SMRIDataset(
        csv_file="va_9194_rep_4.csv",
        root_dir="/home/users/ybi3/SMLvsDL/SampleSplits_Sex/"
    )

    test_set = SMRIDataset(
        csv_file="te_9194_rep_4.csv",
        root_dir="/home/users/ybi3/SMLvsDL/SampleSplits_Sex/"
    )
    prefetch_factor = 4
    train_loader = DataLoader(dataset=train_set, batch_size=args.bs, shuffle=True, num_workers=args.nw,
                              drop_last=True, pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=True)
    val_loader = DataLoader(dataset=val_set, batch_size=args.bs, shuffle=True, num_workers=args.nw,
                            drop_last=True, pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.bs, shuffle=True, num_workers=args.nw,
                             drop_last=True, pin_memory=True, prefetch_factor=prefetch_factor, persistent_workers=True)

    return train_loader, val_loader, test_loader

def generate_validation(model):

#    model = initializeNet()

    # Training parameters
    epochs_no_improve = 0

    criterion = nn.CrossEntropyLoss()
    reduce_on = 'max'
    m_val_acc = 0
    history = pd.DataFrame(columns=['scorename', 'epoch',
                                    'tr_acc', 'bal_tr_acc', 'val_acc', 'bal_val_acc', 'loss'])
    criterion.cuda()

    # Declare optimizer
 #   optimizer = optim.Adam(model.parameters(), lr=args.lr)

    optimizer = optim.SGD(model.parameters(),momentum=0.9,lr=args.lr,weight_decay=1e-3)

    # Declare learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode=reduce_on, factor=0.5, patience=7, verbose=True)

    train_loader, val_loader,_ = loaddata()

    for epoch in range(args.es):

        # Train ---------------------------
        print('Training: ')
        loss = train(train_loader,model, criterion, optimizer)
        print('loss in epoch',epoch,'is:',loss)
        writer.add_scalar('loss(rep5 in Pre-Trained SGD)/epoch:', loss, epoch)

       # ---------------------------------------------

        print('Validating: ')

        # Evaluate classification perfromance on training and validation data
        train_acc, bal_train_acc = evalMetrics(train_loader,model)

        print('training accuracy in epoch',epoch,'is:',train_acc,'/ balanced training accuracy:', bal_train_acc)

        writer.add_scalar('train_acc(rep5 in Pre-Trained SGD)/epochs: ', train_acc, epoch)
        writer.add_scalar('bal_train_acc(rep5 in Pre-Trained SGD)/epochs: ', bal_train_acc,epoch)

        valid_acc, bal_valid_acc = evalMetrics(val_loader,model)

        print('validation accuracy in epoch', epoch, 'is:', valid_acc, '/ balanced validation accuracy:', bal_valid_acc)

        writer.add_scalar('val_acc(rep5 in Pre-Trained SGD)/epochs: ', valid_acc,epoch)
        writer.add_scalar('bal_valid_acc(rep5 in Pre-Trained SGD)/epochs: ',bal_valid_acc,epoch)

      #  -------------------------------------------------------

        # Log Performance
        history.loc[epoch] = [args.scorename, epoch, train_acc,
                                  bal_train_acc, valid_acc, bal_valid_acc, loss]

        # Check for maxima (e.g. accuracy for classification)
        isBest = valid_acc > m_val_acc


        # Write Log
        history.to_csv('history_Pre-Trained_SGD5.csv', index=False)

        # Early Stopping
        if args.es_va:

            # If minima/maxima
            if isBest:

                # Reset counter for patience
                epochs_no_improve = 0
                m_val_acc = valid_acc

            else:

                # Update counter for patience
                epochs_no_improve += 1

                # Check early stopping condition
                if epochs_no_improve == args.es_pat:

                    print('Early stopping!')

                    # Stop training: Return to main
                    return history, m_val_acc

        else:
            print('build loss or other cases')

        # Decay Learning Rate
        scheduler.step(valid_acc)

def evaluate_test_accuracy():

    # Load validated net
    model = loadNet()

    device = torch.device("cuda:1")
    model.to(device)
    model.eval()

    # Dataloader
    _,_,testloader = loaddata()


    outs = pd.DataFrame(columns=['acc_te', 'bal_acc_te'])

    print('Testing: ')

    # Evaluate classification performance
    acc, bal_acc = evalMetrics(testloader, model)

    # Log Performance
    outs.loc[0] = [acc, bal_acc]


    # Write Log
    outs.to_csv('test_pretrained_SGD5.csv', index=False)

def read_X_y_5D(df,scorename):
    X,y = [],[]
    for sN in np.arange(df.shape[0]):
        fN = df['smriPath'].iloc[sN]
        la = df[scorename].iloc[sN]
        if scorename == 'label':
            la -= 1
        im = np.float32(nib.load(fN).get_data())
        im = (im - im.min()) / (im.max() - im.min())
        im = np.reshape(im, (1, im.shape[0], im.shape[1], im.shape[2]))
        X.append(im)
        y.append(la)
    X = np.array(X)
    y = np.array(y)
    print('X: ',X.shape,' y: ',y.shape)
    return X, y

def get_avg_saliency():

    path = '/home/users/ybi3/PyC_Project/tmp/pycharm_project_816/temp/'

    total = np.empty((121,145,121))

    for number in range(1000):
        image = np.float32(nib.load(path + 'AO_sex_iter__nSub_' + str(number) + '.nii').get_fdata())
        print(image.shape)

        total = np.add(total, image)

    total = total / 1000

    fname = 'subject_avg.nii'

    nib.save(nib.Nifti1Image(total.squeeze(), np.eye(4)),fname)

def get_gen_svg_saliency(df, scorename):

    path = '/home/users/ybi3/PyC_Project/tmp/pycharm_project_816/temp/'

    indexM = df[df[scorename]=='M'].index.tolist()

    print(type(indexM))

    totalM = np.empty((121,145,121))
    totalF = np.empty((121,145,121))

    for number in range(1000):
        flag = (number in indexM)
        if flag:
            imageM = np.float32(nib.load(path + 'AO_sex_iter__nSub_' + str(number) + '.nii').get_fdata())
            totalM = np.add(totalM, imageM)
        else:
            imageF = np.float32(nib.load(path + 'AO_sex_iter__nSub_' + str(number) + '.nii').get_fdata())
            totalF = np.add(totalF, imageF)

    totalM = totalM / len(indexM)
    totalF = totalF / (1000 - len(indexM))

    fnameM = 'gen_M_avg.nii'
    fnameF = 'gen_F_avg.nii'

    print(totalM.squeeze().shape)

    nib.save(nib.Nifti1Image(totalM.squeeze(), np.eye(4)), fnameM)
    nib.save(nib.Nifti1Image(totalF.squeeze(), np.eye(4)), fnameF)


def load_nifti(file_path, mask=None, z_factor=None, remove_nan=True):
    """Load a 3D array from a NIFTI file."""
    img = nib.load(file_path)
    struct_arr = np.array(img.get_data())

    if remove_nan:
        struct_arr = np.nan_to_num(struct_arr)
    if mask is not None:
        struct_arr *= mask
    if z_factor is not None:
        struct_arr = np.around(zoom(struct_arr, z_factor), 0)

    return struct_arr

def resize_image(img, size, interpolation=0):
    """Resize img to size. Interpolation between 0 (no interpolation) and 5 (maximum interpolation)."""
    zoom_factors = np.asarray(size) / np.asarray(img.shape)
    return sp.ndimage.zoom(img, zoom_factors, order=interpolation)

def get_brain_area_masks(data_size):
    brain_map = load_nifti('/home/users/ybi3/SMLvsDL/aal.nii.gz')
    brain_areas = np.unique(brain_map)[1:]  # omit background

    area_masks = []
    for area in brain_areas:
        area_mask = np.zeros_like(brain_map)
        area_mask[brain_map == area] = 1
        area_mask = resize_image(area_mask, data_size, interpolation=0)
        area_masks.append(area_mask)

    area_names = ['Precentral_L', 'Precentral_R', 'Frontal_Sup_L', 'Frontal_Sup_R', 'Frontal_Sup_Orb_L',
                  'Frontal_Sup_Orb_R', 'Frontal_Mid_L', 'Frontal_Mid_R', 'Frontal_Mid_Orb_L', 'Frontal_Mid_Orb_R',
                  'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L', 'Frontal_Inf_Tri_R',
                  'Frontal_Inf_Orb_L', 'Frontal_Inf_Orb_R', 'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L',
                  'Supp_Motor_Area_R', 'Olfactory_L', 'Olfactory_R', 'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
                  'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R', 'Insula_L', 'Insula_R',
                  'Cingulum_Ant_L', 'Cingulum_Ant_R', 'Cingulum_Mid_L', 'Cingulum_Mid_R', 'Cingulum_Post_L',
                  'Cingulum_Post_R', 'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L', 'ParaHippocampal_R',
                  'Amygdala_L', 'Amygdala_R', 'Calcarine_L', 'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L',
                  'Lingual_R', 'Occipital_Sup_L', 'Occipital_Sup_R', 'Occipital_Mid_L', 'Occipital_Mid_R',
                  'Occipital_Inf_L', 'Occipital_Inf_R', 'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Postcentral_R',
                  'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L', 'Parietal_Inf_R', 'SupraMarginal_L',
                  'SupraMarginal_R', 'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R', 'Paracentral_Lobule_L',
                  'Paracentral_Lobule_R', 'Caudate_L', 'Caudate_R', 'Putamen_L', 'Putamen_R', 'Pallidum_L',
                  'Pallidum_R', 'Thalamus_L', 'Thalamus_R', 'Heschl_L', 'Heschl_R', 'Temporal_Sup_L', 'Temporal_Sup_R',
                  'Temporal_Pole_Sup_L', 'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R',
                  'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L', 'Temporal_Inf_R', 'Cerebelum_Crus1_L',
                  'Cerebelum_Crus1_R', 'Cerebelum_Crus2_L', 'Cerebelum_Crus2_R', 'Cerebelum_3_L', 'Cerebelum_3_R',
                  'Cerebelum_4_5_L', 'Cerebelum_4_5_R', 'Cerebelum_6_L', 'Cerebelum_6_R', 'Cerebelum_7b_L',
                  'Cerebelum_7b_R', 'Cerebelum_8_L', 'Cerebelum_8_R', 'Cerebelum_9_L', 'Cerebelum_9_R',
                  'Cerebelum_10_L', 'Cerebelum_10_R', 'Vermis_1_2', 'Vermis_3', 'Vermis_4_5', 'Vermis_6', 'Vermis_7',
                  'Vermis_8', 'Vermis_9', 'Vermis_10']
    merged_area_names = [name[:-2] for name in area_names[:108:2]] + area_names[108:]

    return area_masks, area_names, merged_area_names

def run_saliency(odir, itrpm, images, net, area_masks, scorename, taskM):
    for nSub in np.arange(images.shape[0]):
        print(nSub)
        fname = odir + itrpm + '_' + scorename + '_iter_'  + '_nSub_' + str(nSub) + '.nii'
        if itrpm == 'AO':
            interpretation_method = area_occlusion
            sal_im = interpretation_method(net, images[nSub], area_masks, occlusion_value=0, apply_softmax=True, cuda=True, verbose=False,taskmode=taskM)
        elif itrpm == 'BP':
            interpretation_method = sensitivity_analysis
            sal_im = interpretation_method(net, images[nSub], apply_softmax=True, cuda=True, verbose=False, taskmode=taskM)
        else:
            print('Verify interpretation method')
        nib.save(nib.Nifti1Image(sal_im.squeeze() , np.eye(4)), fname)


def area_occlusion(model, image_tensor, area_masks, target_class=None, occlusion_value=0, apply_softmax=True,
                   cuda=False, verbose=False, taskmode='clx'):
    image_tensor = torch.Tensor(image_tensor)  # convert numpy or list to tensor

    if cuda:
        image_tensor = image_tensor.cuda()
    output = model(Variable(image_tensor[None], requires_grad=False))[0]

    if apply_softmax:
        output = F.softmax(output)

    if taskmode == 'reg':
        unoccluded_prob = output.data
    elif taskmode == 'clx':
        output_class = output.max(1)[1].data.cpu().numpy()[0]

        if verbose: print('Image was classified as', output_class, 'with probability', output.max(1)[0].data[0])

        if target_class is None:
            target_class = output_class
        unoccluded_prob = output.data[0, target_class]

    relevance_map = torch.zeros(image_tensor.shape[1:])
    if cuda:
        relevance_map = relevance_map.cuda()

    for area_mask in area_masks:

        area_mask = torch.FloatTensor(area_mask)

        if cuda:
            area_mask = area_mask.cuda()
        image_tensor_occluded = image_tensor * (1 - area_mask).view(image_tensor.shape)

        output = model(Variable(image_tensor_occluded[None], requires_grad=False))[0]
        if apply_softmax:
            output = F.softmax(output)

        if taskmode == 'reg':
            occluded_prob = output.data
        elif taskmode == 'clx':
            occluded_prob = output.data[0, target_class]

        ins = area_mask.view(image_tensor.shape) == 1
        ins = ins.squeeze()
        relevance_map[ins] = (unoccluded_prob - occluded_prob)

    relevance_map = relevance_map.cpu().numpy()
    relevance_map = np.maximum(relevance_map, 0)
    return relevance_map


def sensitivity_analysis(model, image_tensor, target_class=None, postprocess='abs', apply_softmax=True, cuda=False,
                         verbose=False, taskmode='clx'):
    # Adapted from http://arxiv.org/abs/1808.02874
    # https://github.com/jrieke/cnn-interpretability

    if postprocess not in [None, 'abs', 'square']:
        raise ValueError("postprocess must be None, 'abs' or 'square'")

    # Forward pass.
    image_tensor = torch.Tensor(image_tensor)  # convert numpy or list to tensor        print(image_tensor.shape)

    if cuda:
        image_tensor = image_tensor.cuda()
    X = Variable(image_tensor[None], requires_grad=True)  # add dimension to simulate batch

    output = model(X)[0]

    if apply_softmax:
        output = F.softmax(output)

    # print(output.shape)

    # Backward pass.
    model.zero_grad()

    if taskmode == 'reg':
        output.backward(gradient=output)
    elif taskmode == 'clx':
        output_class = output.max(1)[1].data[0]
        if verbose: print('Image was classified as', output_class, 'with probability', output.max(1)[0].data[0])
        one_hot_output = torch.zeros(output.size())
        if target_class is None:
            one_hot_output[0, output_class] = 1
        else:
            one_hot_output[0, target_class] = 1
        if cuda:
            one_hot_output = one_hot_output.cuda()
        output.backward(gradient=one_hot_output)

    relevance_map = X.grad.data[0].cpu().numpy()

    # Postprocess the relevance map.
    if postprocess == 'abs':  # as in Simonyan et al. (2014)
        return np.abs(relevance_map)
    elif postprocess == 'square':  # as in Montavon et al. (2018)
        return relevance_map ** 2
    elif postprocess is None:
        return relevance_map