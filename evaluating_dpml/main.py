import re
import sys, datetime
import os
import argparse
import pickle
import numpy as np

from evaluatingDPML.core.classifier import get_predictions

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from core.attack import save_data
from core.attack import load_data
from core.attack import train_target_model
from core.attack import yeom_membership_inference
from core.attack import shokri_membership_inference
from core.attack import yeom_attribute_inference
from core.utilities import log_loss
from core.utilities import get_random_features
from sklearn.metrics import roc_curve
from scipy.spatial import distance as ssdistance

RESULT_PATH = f'results/'

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)


def getClosestTo(base:np.ndarray, domain:np.ndarray):

    return np.argmin(ssdistance.cdist(base, domain), axis=1)

def isDiagonal(arr:np.ndarray):
    return arr==np.arange(len(arr))

def _getScore(real, pred):
    fpr, tpr, thresholds = roc_curve(real, pred, pos_label=1)
    return tpr[1]-fpr[1]

def getScore(real, pred, threshold = 1, invert = False):
    if invert:
        pred = -pred
    fpr, tpr, thresholds = roc_curve(real, pred, pos_label=1)
    score = tpr-fpr
    if invert:
        threshold = -threshold
    if threshold is None:
        return score, -thresholds
    for thr, sc in reversed(list(zip(thresholds,score))):
        if thr>=threshold:
            return sc
    return 0

def getResultDirName(args):
    if args.target_privacy == 'no_privacy':
        return f'{args.target_model}_no_privacy_{str(args.target_l2_ratio)}_{str(args.run)}'
    else:
        return f'{args.target_model}_{args.target_privacy}_{args.target_dp}_{str(args.target_epsilon)}_{str(args.run)}'

def run_experiment(args, run_anyway=False,save_model = False,**kwargs):
    isMethodBased = args.get('method',None) is not None

    if run_anyway:
        if os.path.exists(os.path.join(RESULT_PATH, args.train_dataset, getResultDirName(args), 'results.p')):
            print('Results already found for', args)
            return

    results = {}

    print('-' * 10 + 'TRAIN TARGET' + '-' * 10 + '\n')
    dataset = load_data('target_data.npz', args)
    train_x, train_y, test_x, test_y = dataset
    true_x = np.vstack((train_x, test_x))
    true_y = np.append(train_y, test_y)
    batch_size = args.target_batch_size

    pred_y, membership, test_classes, classifier, aux = train_target_model(
        args=args,
        dataset=dataset,
        epochs=args.target_epochs,
        batch_size=args.target_batch_size,
        learning_rate=args.target_learning_rate,
        clipping_threshold=args.target_clipping_threshold,
        n_hidden=args.target_n_hidden,
        l2_ratio=args.target_l2_ratio,
        model=args.target_model,
        privacy=args.target_privacy,
        dp=args.target_dp,
        epsilon=args.target_epsilon,
        delta=args.target_delta,
        save=args.save_model)
    train_loss, train_acc, test_loss, test_acc = aux
    per_instance_loss = np.array(log_loss(true_y, pred_y))

    import tensorflow as tf
    assert isinstance(classifier,tf.estimator.Estimator)

    if isMethodBased:
        from DPMLadapter.ArgsObject import ArgsObject
        assert isinstance(args,ArgsObject)
        real_train_x, real_train_y, real_test_x, real_test_y = load_data('target_data.npz',args.withChange(train_dataset=args['base_dataset']))
        real_true_x = np.vstack((real_train_x, real_test_x))
        real_true_y = np.append(real_train_y, real_test_y)
        assert (real_true_y==true_y).all()
        #Recreate train_loss

        # from classifier.py at 185
        import tensorflow as tf

        train_eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x={'x': real_train_x},
            y= real_train_y,
            num_epochs=1,
            shuffle=False)
        # From classifier.py at 219
        eval_results = classifier.evaluate(input_fn=train_eval_input_fn)
        real_train_loss = eval_results['loss']
        real_train_acc = eval_results['accuracy']
        print('Train accuracy is: %.3f' % (real_train_acc))
        del eval_results

        test_eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x = {'x': real_test_x},
            y = real_test_y,
            num_epochs=1,
            shuffle=False)
        eval_results = classifier.evaluate(input_fn=test_eval_input_fn)
        real_test_loss = eval_results['loss']
        real_test_acc = eval_results['accuracy']
        print('Test accuracy is: %.3f' % (real_test_acc))
        del eval_results

        real_AUX = (real_train_loss, real_train_acc, real_test_loss, real_test_acc)
        #test_acc = real_test_acc

        real_pred_y = []
        predictions = classifier.predict(input_fn=train_eval_input_fn)
        _, pred_scores = get_predictions(predictions)
        real_pred_y.append(pred_scores)
        del predictions, pred_scores

        predictions = classifier.predict(input_fn=test_eval_input_fn)
        _, pred_scores = get_predictions(predictions)
        real_pred_y.append(pred_scores)
        del predictions, pred_scores

        real_pred_y = np.vstack(real_pred_y)
        real_pred_y = real_pred_y.astype('float32')

        real_per_instance_loss = np.array(log_loss(real_true_y, real_pred_y))
    else:
        realAUX = aux
        real_train_loss, real_train_acc, real_test_loss, real_test_acc = train_loss, train_acc, test_loss, test_acc
        real_train_x, real_train_y, real_test_x, real_test_y = train_x, train_y, test_x, test_y
        real_pred_y = pred_y
        real_per_instance_loss = per_instance_loss
        real_true_x = true_x
        real_true_y = true_y

    '''if genFeatures:
        features = get_random_features(true_x, range(true_x.shape[1]), 5)
        print(features)'''
    '''
    if yeom_mi and False:
        # Yeom's membership inference attack when only train_loss is known
        pred_membership = yeom_membership_inference(per_instance_loss, membership, train_loss)

        fpr, tpr, thresholds = roc_curve(membership, pred_membership, pos_label=1)
        yeom_mem_adv = tpr[1] - fpr[1]
        results['yeom_mi'] = {'score':yeom_mem_adv,'prediction':pred_membership}

        if isMethodBased:
            distances = ssdistance.cdist(real_true_x, true_x)
            # TODO version with real_train_loss

            real_pred_membership:np.ndarray = yeom_membership_inference(real_per_instance_loss, membership, real_train_loss)

            #fpr, tpr, thresholds = roc_curve(membership, real_pred_membership, pos_label=1)
            #real_yeom_mem_adv = tpr[1] - fpr[1]
            real_yeom_mem_adv = getScore(membership,real_pred_membership)
            results['real_yeom_mi'] = {'score':real_yeom_mem_adv,'prediction':real_pred_membership}

            linkage = np.argmax(distances, axis=0) # find the closes original point to the perturbed
            adv2_pred_membership = np.zeros(real_pred_membership.shape)
            adv2_pred_membership[linkage[np.where(real_pred_membership)]] = 1
            adv2_yeom_mem_adv = getScore(membership,adv2_pred_membership)
            #fpr, tpr, thresholds = roc_curve(membership, adv2_pred_membership, pos_label=1)
            #adv2_yeom_mem_adv = tpr[1] - fpr[1]

            results['yeom_mi_adv2'] = {'score':adv2_yeom_mem_adv,'prediction':adv2_pred_membership}
            closest_dist = np.min(distances[:,pred_membership.astype(bool)],axis=1)


            max_score = np.max(getScore(membership,closest_dist,threshold=None,invert=True)[0])
            results['improved_yeom_mi'] = {'score':getScore(membership,(closest_dist<=np.median(closest_dist)).astype(int)),'max_score':max_score,'prediction':real_pred_membership}
    '''
    results['membership'] = membership

    results['train_acc'] = train_acc
    results['test_acc'] = test_acc
    results['train_loss'] = train_loss
    results['test_loss'] = test_loss
    results['per_instance_loss'] = per_instance_loss
    results['pred_y'] = pred_y

    results['real_train_acc'] = real_train_acc
    results['real_test_acc'] = real_test_acc
    results['real_train_loss'] = real_train_loss
    results['real_test_loss'] = real_test_loss
    results['real_per_instance_loss'] = real_per_instance_loss
    results['real_pred_y'] = real_pred_y

    results['args'] = dict(args)

    '''if shokri_mi:
        assert retrain, 'To reevaluate shokri_mi the classifier must be retrained.'
        # Shokri's membership inference attack based on shadow model training
        shokri_mi_outputs = shokri_membership_inference(args, pred_y, membership, test_classes)
        shokri_mem_adv, _, shokri_mem_confidence, _, _, _, _ = shokri_mi_outputs

    if yeom_ai:
        assert retrain, 'To reevaluate yeom_ai the classifier must be retrained.'
        # Yeom's attribute inference attack when train_loss is known - Adversary 4 of Yeom et al.
        pred_membership_all, true_attribute_value_all, pred_attribute_value_all = yeom_attribute_inference(true_x, true_y, classifier, membership, features, train_loss)
        yeom_attr_adv = []
        for pred_membership,true_attribute_value,pred_attribute_value in zip(pred_membership_all,true_attribute_value_all,pred_attribute_value_all):
            fpr, tpr, thresholds = roc_curve(true_attribute_value, pred_attribute_value, pos_label=1)
            yeom_attr_adv.append(tpr[1] - fpr[1])
        print('-'*10)
        print(yeom_attr_adv)
        print('-'*10)'''


    '''
    if args.target_privacy == 'no_privacy':
        #pickle.dump([train_acc, test_acc, train_loss, membership, shokri_mem_adv, shokri_mem_confidence, yeom_mem_adv, per_instance_loss, yeom_attr_adv, pred_membership_all, features], open(RESULT_PATH+args.train_dataset+'/'+args.target_model+'_no_privacy_'+str(args.target_l2_ratio)+'.p', 'wb'))
        pickle.dump([train_acc, test_acc, train_loss, membership, shokri_mem_adv, shokri_mem_confidence, yeom_mem_adv, per_instance_loss, yeom_attr_adv, pred_membership_all, features], open(RESULT_PATH+args.train_dataset+'/'+args.target_model+'_no_privacy_'+str(args.target_l2_ratio)+'_'+str(args.run)+'.p', 'wb'))
    else:
        pickle.dump([train_acc, test_acc, train_loss, membership, shokri_mem_adv, shokri_mem_confidence, yeom_mem_adv, per_instance_loss, yeom_attr_adv, pred_membership_all, features], open(RESULT_PATH+args.train_dataset+'/'+args.target_model+'_'+args.target_privacy+'_'+args.target_dp+'_'+str(args.target_epsilon)+'_'+str(args.run)+'.p', 'wb'))
    '''


    resultDir = getResultDirName(args)
    if not os.path.exists(os.path.join(RESULT_PATH, args.train_dataset)):
        os.makedirs(os.path.join(RESULT_PATH, args.train_dataset))
    if not os.path.exists(os.path.join(RESULT_PATH, args.train_dataset,resultDir)):
        os.makedirs(os.path.join(RESULT_PATH, args.train_dataset,resultDir))

    if save_model:
        def serving_input_fn():
            inputs = {'x': tf.compat.v1.placeholder(tf.float32, [None, train_x.shape[1]])}
            return tf.estimator.export.ServingInputReceiver(inputs, inputs)

        save_path = classifier.export_saved_model(os.path.join(RESULT_PATH, args.train_dataset, resultDir), serving_input_fn).decode()
        results['model_dirname'] = os.path.basename(save_path)

    '''import tensorflow.compat.v1 as tf1
    loaded = tf.saved_model.load(save_path)

    class LayerFromSavedModel(tf1.keras.layers.Layer):
        def __init__(self):
            super(LayerFromSavedModel, self).__init__()
            self.vars = loaded.variables

        def call(self, inputs):
            return loaded.signatures['serving_default'](inputs)

    input = tf1.keras.Input(shape=(loaded.variables.variables[0].shape[0],))
    importedModel = tf1.keras.Model(input, LayerFromSavedModel()(input))

    predictions = importedModel.predict(true_x)
    _, check_pred_y = get_predictions(predictions)
    del predictions

    check_true_y = check_pred_y.astype('float32')
    assert np.isclose(pred_y,check_pred_y).all()'''

    pickle.dump(results, open(os.path.join(RESULT_PATH, args.train_dataset,resultDir,'results.p'),'wb'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dataset', type=str)
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--use_cpu', type=int, default=0)
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--save_data', type=int, default=0)
    # target and shadow model configuration
    parser.add_argument('--n_shadow', type=int, default=5)
    parser.add_argument('--target_data_size', type=int, default=int(1e4))
    parser.add_argument('--target_test_train_ratio', type=int, default=1)
    parser.add_argument('--target_model', type=str, default='nn')
    parser.add_argument('--target_learning_rate', type=float, default=0.01)
    parser.add_argument('--target_batch_size', type=int, default=200)
    parser.add_argument('--target_n_hidden', type=int, default=256)
    parser.add_argument('--target_epochs', type=int, default=100)
    parser.add_argument('--target_l2_ratio', type=float, default=1e-8)
    parser.add_argument('--target_clipping_threshold', type=float, default=1)
    parser.add_argument('--target_privacy', type=str, default='no_privacy')
    parser.add_argument('--target_dp', type=str, default='dp')
    parser.add_argument('--target_epsilon', type=float, default=0.5)
    parser.add_argument('--target_delta', type=float, default=1e-5)
    # attack model configuration
    parser.add_argument('--attack_model', type=str, default='nn')
    parser.add_argument('--attack_learning_rate', type=float, default=0.01)
    parser.add_argument('--attack_batch_size', type=int, default=100)
    parser.add_argument('--attack_n_hidden', type=int, default=64)
    parser.add_argument('--attack_epochs', type=int, default=100)
    parser.add_argument('--attack_l2_ratio', type=float, default=1e-6)

    # parse configuration
    args = parser.parse_args()
    print(vars(args))
    
    # Flag to disable GPU
    if args.use_cpu:
    	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

    if args.save_data:
        save_data(args)
    else:
        run_experiment(args)