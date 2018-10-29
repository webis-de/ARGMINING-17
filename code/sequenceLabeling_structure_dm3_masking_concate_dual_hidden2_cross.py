#!/usr/bin/env python
# -*- coding: utf-8 -*-

# use recurrent NN for sequencial labeling task
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.preprocessing import sequence
from keras.layers import Embedding, Input, merge, Dropout, recurrent, TimeDistributed, Bidirectional, concatenate
from keras.layers.core import Activation, Dense, Flatten, Lambda, Reshape, Dense, Masking
import numpy as np 
from keras.layers.convolutional import Convolution1D, AveragePooling1D, MaxPooling1D
from keras.utils.np_utils import to_categorical

from keras.optimizers import Adagrad,SGD,Adam,RMSprop
from sklearn.metrics import precision_recall_fscore_support
from keras import backend as K
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint

import sys,time,codecs, pickle, os, random, os.path, json
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import f1_score,precision_recall_fscore_support,log_loss

import public_library as lib

_EPSILON = 10e-8

if __name__ == '__main__':
    pickleDir = '../../../pickle/'
    h5Dir = '../../../h5/'
    weightDir = '../../../weight/'
    modelDir = '../../../model/'
    
    t1 = time.time()

# choose pre-trained embedding
#     source = 'google'
    source = 'glove'

# choose merging method
#     merging = 'max'
#     merging = 'ave'
#     merging = 'concat'
    merging = 'hidden'
#     merging = 'transform'
    
    opt = 'rms'
    
    if source == 'google':
        embeddingFile = '../../../../embeddings/GoogleNews-vectors-negative300.txt'
        vDim = 300
    else:
        embeddingFile = '../../../../embeddings/glove.6B.300d.txt'
        vDim = 300

    
    voc,embeddings = lib.loadEmbedding(embeddingFile,vDim)
    print 'Load Embedding Elapse: ', time.time()-t1
    vSize = len(embeddings)
    
    t2 = time.time()

    patience = 10
    maxEpoch = 200
    batchSize = 8
    dataDir = '../../../data/features/'
    
    lDic = lib.genLDic()
    RNN = recurrent.LSTM
    
    if merging == 'transform':
        trainable = False
    else:
        trainable = True
    
    devSplit = 0.1 # select 10% from training data as development data
    
    train_target = 'essays'
#     train_target = 'editorials'
#     train_target = 'webDiscourse'

    test_target = 'essays'
#     test_target = 'editorials'
#     test_target = 'webDiscourse'
    
    splitFiles = [dataDir+train_target+'/train.lst',dataDir+test_target+'/test.lst']
    [trainlist, devlist, testlist] = lib.getSplitDataFile(splitFiles, devSplit)
       
    features = ['emb','str']
    

    
    [llistTrain, wlistTrain, slistTrain, clistTrain, plistTrain, tlistTrain, dmlistTrain, pmlistTrain] = lib.loadList(dataDir+'cross-domain/simple/'+train_target+'-split/testingSet/',trainlist,'_'.join(features))
    [llistDev, wlistDev, slistDev, clistDev, plistDev, tlistDev, dmlistDev, pmlistDev] = lib.loadList(dataDir+'cross-domain/simple/'+train_target+'-split/testingSet/',devlist,'_'.join(features))
    [llistTest, wlistTest, slistTest, clistTest, plistTest, tlistTest, dmlistTest, pmlistTest] = lib.loadList(dataDir+'cross-domain/simple/'+test_target+'-split/testingSet/',testlist,'_'.join(features))
     
    clistTrain = lib.convert2BIE(clistTrain)
    clistDev = lib.convert2BIE(clistDev)
    clistTest = lib.convert2BIE(clistTest)
    
    plistTrain = lib.convert2BIE(plistTrain)
    plistDev = lib.convert2BIE(plistDev)
    plistTest = lib.convert2BIE(plistTest)

    dmlistTrain = lib.locateDM(dmlistTrain,wlistTrain,slistTrain,trainlist)
    dmlistDev = lib.locateDM(dmlistDev,wlistDev,slistDev)
    dmlistTest = lib.locateDM(dmlistTest,wlistTest,slistTest)
    
    sDic = lib.genSCTDic(slistTrain)
    cDic = lib.genSCTDic(clistTrain)
    pDic = lib.genSCTDic(plistTrain)
    tDic = lib.genSCTDic(tlistTrain)
    dmDic = lib.genSCTDic(dmlistTrain)
    pmDic = lib.genSCTDic(pmlistTrain)
    
    if merging != 'transform':
        sDim = vDim
        cDim = vDim
        pDim = vDim
        tDim = vDim
        dmDim = vDim
        pmDim = vDim
    else:
        rDim = vDim
        sDim = vDim * rDim
        cDim = vDim * rDim
        tDim = vDim * rDim
        dmDim = vDim * rDim
        pmDim = vDim * rDim
        
    sSize = len(sDic)
    cSize = len(cDic)
    pSize = len(pDic)
    tSize = len(tDic)
    dmSize = len(dmDic)
    pmSize = len(pmDic)
    
    t3 = time.time()
    print 'Load Data Elapse: ', t3-t2    
    
    rSize = 128 # recurrent hidden layer size
    lSize = len(lDic)   # number of labels
    
    maxFeatureLength,maxTotalLength = lib.getMaxDoc([wlistTrain,wlistDev,wlistTest])

    
    samples = len(wlistTrain)
    samplesDev = len(wlistDev)
    samplesTest = len(wlistTest)
    
    print 'Sizes:',samples,samplesDev,samplesTest
    print 'maxDoc:',maxTotalLength
    print 'maxFeature:',maxFeatureLength
    
    rndBase = 0.01
    
    inputW = lib.dicLookUp(wlistTrain,voc,samples,maxTotalLength,maxFeatureLength)   
    inputWDev = lib.dicLookUp(wlistDev,voc,samplesDev,maxTotalLength,maxFeatureLength)  
    inputWTest = lib.dicLookUp(wlistTest,voc,samplesTest,maxTotalLength,maxFeatureLength)  
    
    inputS = lib.genHiddenInputSCT(slistTrain,sDic,samples,maxTotalLength)
    inputSDev = lib.genHiddenInputSCT(slistDev,sDic,samplesDev,maxTotalLength)
    inputSTest = lib.genHiddenInputSCT(slistTest,sDic,samplesTest,maxTotalLength)
    
    inputC = lib.genHiddenInputSCT(clistTrain,cDic,samples,maxTotalLength)
    inputCDev = lib.genHiddenInputSCT(clistDev,cDic,samplesDev,maxTotalLength)
    inputCTest = lib.genHiddenInputSCT(clistTest,cDic,samplesTest,maxTotalLength)
    
    inputP = lib.genHiddenInputSCT(plistTrain,pDic,samples,maxTotalLength)
    inputPDev = lib.genHiddenInputSCT(plistDev,pDic,samplesDev,maxTotalLength)
    inputPTest = lib.genHiddenInputSCT(plistTest,pDic,samplesTest,maxTotalLength)
    
    inputT = lib.genHiddenInputSCT(tlistTrain,tDic,samples,maxTotalLength)
    inputTDev = lib.genHiddenInputSCT(tlistDev,tDic,samplesDev,maxTotalLength)
    inputTTest = lib.genHiddenInputSCT(tlistTest,tDic,samplesTest,maxTotalLength)
    
    inputDM = lib.genHiddenInputSCT(dmlistTrain,dmDic,samples,maxTotalLength)
    inputDMDev = lib.genHiddenInputSCT(dmlistDev,dmDic,samplesDev,maxTotalLength)
    inputDMTest = lib.genHiddenInputSCT(dmlistTest,dmDic,samplesTest,maxTotalLength)
        
    inputL = lib.labelLookUp(llistTrain,lDic,samples,maxTotalLength,lSize)
    inputLDev = lib.labelLookUp(llistDev,lDic,samplesDev,maxTotalLength,lSize)
    inputLTest = lib.labelLookUp(llistTest,lDic,samplesTest,maxTotalLength,lSize)
        
    print 'data shape:',inputW.shape
    print 'data shape:',inputS.shape
    print 'label shape:',inputL.shape
    
    
    inputWord = Input(shape=(maxTotalLength*maxFeatureLength,),dtype='int32',name='inputWord')
    if merging == 'hidden': 
        inputSen = Input(shape=(maxTotalLength,len(sDic)),dtype='float32',name='inputSen') 
        inputCla = Input(shape=(maxTotalLength,len(cDic)),dtype='float32',name='inputCla') 
        inputPhr = Input(shape=(maxTotalLength,len(pDic)),dtype='float32',name='inputPhr') 
        inputTok = Input(shape=(maxTotalLength,len(tDic)),dtype='float32',name='inputTok')
        inputDis = Input(shape=(maxTotalLength,len(dmDic)),dtype='float32',name='inputDis')
    else:
        inputSen = Input(shape=(maxTotalLength,),dtype='int32',name='inputSen') 
        inputCla = Input(shape=(maxTotalLength,),dtype='int32',name='inputCla') 
        inputPhr = Input(shape=(maxTotalLength,),dtype='int32',name='inputPhr') 
        inputTok = Input(shape=(maxTotalLength,),dtype='int32',name='inputTok')
        inputDis = Input(shape=(maxTotalLength,),dtype='int32',name='inputDis')

    embeddingW = Embedding(output_dim=vDim,input_dim=vSize,input_length=maxTotalLength*maxFeatureLength,weights=[embeddings], trainable=trainable, mask_zero=True)(inputWord)
  
#     embeddingW = Embedding(output_dim=vDim,input_dim=vSize,input_length=maxTotalLength*maxFeatureLength,weights=[embeddings], mask_zero=True, trainable=trainable)(inputWord)
#     embeddingS = Embedding(output_dim=sDim,input_dim=sSize,input_length=maxTotalLength, mask_zero=True)(inputSen)
#     embeddingC = Embedding(output_dim=cDim,input_dim=cSize,input_length=maxTotalLength, mask_zero=True)(inputCla)
#     embeddingP = Embedding(output_dim=pDim,input_dim=pSize,input_length=maxTotalLength, mask_zero=True)(inputPhr)
#     embeddingT = Embedding(output_dim=tDim,input_dim=tSize,input_length=maxTotalLength, mask_zero=True)(inputTok)
#     embeddingDM = Embedding(output_dim=dmDim,input_dim=dmSize,input_length=maxTotalLength, mask_zero=True)(inputDis)

    if merging == 'max':
        SW_concate = merge([embeddingW,embeddingS],mode='concat',concat_axis=1)
        SW_pooled = MaxPooling1D(pool_length=2,border_mode='valid')(SW_concate)
        SW = Reshape((maxTotalLength,vDim))(SW_pooled)
    elif merging == 'transform':
        RS = Reshape((vDim,rDim))
        SWs=[]
        for i in range(maxTotalLength):
            Wi = Lambda(lib.getWrapper(i),output_shape=lib.getWrapper_output_shape)(embeddingW)
            Si = Lambda(lib.getWrapper(i),output_shape=lib.getWrapper_output_shape)(embeddingS)
            Ci = Lambda(lib.getWrapper(i),output_shape=lib.getWrapper_output_shape)(embeddingC)
            Ti = Lambda(lib.getWrapper(i),output_shape=lib.getWrapper_output_shape)(embeddingT)
            
            SiReshape = RS(Si)
            CiReshape = RS(Ci)
            TiReshape = RS(Ti)
            
            SWi = merge([Wi,SiReshape],mode='dot',dot_axes=(1,1))
            CWi = merge([SWi,CiReshape],mode='dot',dot_axes=(1,1))
            TWi = merge([CWi,TiReshape],mode='dot',dot_axes=(1,1))
            
            SWs.append(TWi)
        SW_merge = merge(SWs,mode='concat',concat_axis=1)
        SW = Reshape((maxTotalLength,rDim))(SW_merge)
    elif merging == 'hidden':
        SW = embeddingW
    else:
        SW = concatenate([embeddingW,embeddingS,embeddingC,embeddingP,embeddingT,embeddingDM])

    if merging == 'hidden':
        rnn_w = Bidirectional(RNN(rSize,return_sequences=True,dropout=0.2,recurrent_dropout=0.2))(SW)
        extra = concatenate([inputSen,inputCla,inputPhr,inputTok,inputDis])
        rnn_e = Bidirectional(RNN(rSize,return_sequences=True,dropout=0.2,recurrent_dropout=0.2))(extra)
        rnn = concatenate([rnn_w,rnn_e])
    else:
        rnn = Bidirectional(RNN(rSize,return_sequences=True,dropout=0.2,recurrent_dropout=0.2))(SW)
    
    predict = TimeDistributed(Dense(lSize,activation='softmax'))(rnn)
    

    rnn2 = Bidirectional(RNN(rSize,return_sequences=True))(predict)    
    predict2 = TimeDistributed(Dense(lSize,activation='softmax'))(rnn2)
    
    model = Model(inputs=[inputWord,inputSen,inputCla,inputPhr,inputTok,inputDis],outputs=[predict,predict2])

    if opt == 'rms':
        optimizer = RMSprop() # optimizer
    else:
        optimizer = Adam() # optimizer
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])

    stop = EarlyStopping(monitor='val_acc',patience=patience,mode='max')
    save = ModelCheckpoint(weightDir + 'LSTM_' + '_'.join(features) + '_'  + 'bidirection2_3Layer_vertical_dm3(BEST_BIE)_' + merging + '_' + str(vDim) + '_' + opt + '.h5', monitor='val_acc',save_best_only=True,mode='max',save_weights_only=True)
    model.fit([inputW,inputS,inputC,inputT,inputDM],inputL,callbacks=[stop,save],batch_size=batchSize,nb_epoch=maxEpoch,validation_data=([inputWDev,inputSDev,inputCDev,inputTDev,inputDMDev],inputLDev))


    ignore = [0]
     
    devG = lib.predictFromClass(inputLDev,ignore)
    dev = lib.filterLabel(devG,devG,ignore)
      
    devDis = lib.filterDisLabel(inputLDev,ignore)
    devDis = lib.filterDisPadding(devDis,devG,ignore)
    max_f1 = 0
    min_loss = 1
    pat = 0
    for epoch in range(maxEpoch):
        model.fit([inputW,inputS,inputC,inputP,inputT,inputDM],[inputL,inputL],batch_size=batchSize,epochs=1)
        [classes_pre,classes] = model.predict([inputWDev,inputSDev,inputCDev,inputPDev,inputTDev,inputDMDev], batch_size=128)
        prediction = lib.predictFromClass(classes,ignore)
        prediction = lib.filterLabel(prediction,devG,ignore)
        f1 = f1_score(dev,prediction, average='macro')    
        acc = accuracy_score(dev,prediction)
             
        preDis = lib.filterDisLabel(classes,ignore)
        preDis = lib.filterDisPadding(preDis,devG,ignore)
             
        loss = log_loss(devDis,preDis)
        if (abs(max_f1 - f1) < _EPSILON and loss <= min_loss) or f1 > max_f1:
            max_f1 = f1
            min_loss = loss
            pat = 0
            model.save_weights(weightDir + target + '_LSTM_masking' + '_'.join(features) + '_'  + 'bidirection_3Layer_vertical_SCP_dm3_dual(BEST!_BIE_' + source + '_dropout2_0)_' + merging + '_' + str(vDim) + '_' + opt + '.h5', True)
        else:
            pat += 1
        print 'epoch:',epoch,'monitor:',f1,'acc:',acc,'loss:',loss,'patience:',pat
        if pat >= patience:
            break

    model.load_weights(weightDir + train_target + '_LSTM_masking' + '_'.join(features) + '_'  + 'bidirection_3Layer_vertical_SCP_dm3_dual(BEST!_BIE_' + source + '_dropout2_0)_' + merging + '_' + str(vDim) + '_' + opt + '.h5')    
    
    [classes_pre,classes] = model.predict([inputWTest,inputSTest,inputCTest,inputPTest,inputTTest,inputDMTest], batch_size=128)
    
    gold = lib.predictFromClass(inputLTest,ignore)
    prediction = lib.predictFromClass(classes,ignore)
    prediction = lib.filterLabel(prediction,gold,ignore)
    conflict = lib.BIOconflict(prediction)
    gold = lib.filterLabel(gold,gold,ignore)
    
    f1 = f1_score(gold,prediction, average=None)
    pr = precision_recall_fscore_support(gold,prediction, average='macro')
    print 'Acc:',accuracy_score(gold,prediction)
    print 'By-class F1 (B, I, O):', f1[0],f1[1],f1[2]
    print 'Macro PR:', pr[0],pr[1]
    print 'Macro F1:',f1_score(gold,prediction, average='macro')
    print 'conflict:',conflict
    
    nFold = 5
    [golds,predictions] = lib.randomFolds(gold,prediction,nFold)
    f1s = []
    for i in range(nFold):
        f1s.append(f1_score(golds[i],predictions[i], average='macro'))
    
    print f1s
    print lib.sigTestStab(f1s)

