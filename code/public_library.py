import numpy as np 
from sklearn.metrics import precision_recall_fscore_support

import sys,time,codecs, pickle, os, random, os.path, json
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import f1_score,precision_recall_fscore_support
import copy
from sklearn.preprocessing import normalize
from scipy.stats import ttest_1samp
import itertools 

_EPSILON = 10e-8

def getWrapper(i):
    def getCmt(x):
        return x[:,i,:]
    return getCmt

def getWrapper_output_shape(input_shape):
    shape = list(input_shape)
    return [shape[0]]+shape[2:]

def maxmerge(x):
    s = K.max(x,axis=0)
    return s
    
def maxmerge_shape(input_shape):
    shapes = list(input_shape)
    return shapes[0]

def loadEmbedding(Embfile,dim):
    voc = {}
    print 'Loading:',Embfile
    with open(Embfile) as f:
        lines = [line.rstrip('\n') for line in f]
    emb=np.zeros((len(lines),dim))    # note: add a "zero" embedding for zero padding
    for l, line in enumerate(lines):
        tokens = line.split(' ')
        emb[l]= [float(t) for t in tokens[1:]]
        voc[tokens[0]] = l+2

    average = np.average(emb, axis=0)
    zero = np.zeros(dim)
    emb = np.append([zero,average],emb,axis=0)
    voc['<zero>'] = 0
    voc['<unk>'] = 1    
    return voc,emb

def loadList(dir,filelist,feature):
    llist = []
    wlist = []
    slist = []
    clist = []
    plist = []
    tlist = []
    dmlist = []
    pmlist = []
    
    for file in filelist:
        filename = dir + file
        if os.path.isfile(filename):
            with open(filename,'r') as f:
                lines = [line.rstrip('\n') for line in f]
                ll = []
                ww = []
                ss = []
                cc = []
                pp = []
                tt = []
                dmm = []
                pmm = [] 
                for line in lines:
                    try:
                        tokens = line.split('\t\t')
                        ll.append(tokens[0])
                        w = [tokens[1].lower()]
                        ww.append(w)
                        p = tokens[2]
                        pp.append(p)
                        c = tokens[3]
                        cc.append(c)
                        s = tokens[4]
                        ss.append(s)
                        t = tokens[5]
                        tt.append(t)
                        dm = tokens[6]
                        dmm.append(dm)
                        pm = tokens[7]
                        pmm.append(pm)
                    except:
                        print filename
                        continue
                    
                llist.append(ll)
                wlist.append(ww)
                slist.append(ss)
                clist.append(cc)
                plist.append(pp)
                tlist.append(tt)
                dmlist.append(dmm)
                pmlist.append(pmm)
                
    return llist, wlist, slist, clist, plist, tlist, dmlist, pmlist

def convert2twoway(llist, wlist, slist, clist, tlist, dlist):
    llistOut = []
    wlistOut = []
    slistOut = []
    clistOut = []
    tlistOut = []
    dlistOut = []
    
    wlistOutL = []
    slistOutL = []
    clistOutL = []
    tlistOutL = []
    dlistOutL = []
    
    wlistOutR = []
    slistOutR = []
    clistOutR = []
    tlistOutR = []
    dlistOutR = []
    
    for idoc, doc in enumerate(llist):
        for itok, token in enumerate(doc):
            llistOut.append(token)
            wlistOut.append(wlist[idoc][itok])   # only one token
            clistOut.append([clist[idoc][itok]])
            tlistOut.append([tlist[idoc][itok]])
            dlistOut.append([dlist[idoc][itok]])
            
            wlistOutL.append(flatten2Dlist(wlist[idoc][:itok]))
            wlistOutR.append(flatten2Dlist(wlist[idoc][itok+1:]))
            
            clistOutL.append(clist[idoc][:itok])
            clistOutR.append(clist[idoc][itok+1:])
            
            tlistOutL.append(tlist[idoc][:itok])
            tlistOutR.append(tlist[idoc][itok+1:])
            
            dlistOutL.append(dlist[idoc][:itok])
            dlistOutR.append(dlist[idoc][itok+1:])
               
    return llistOut, wlistOut, slistOut, clistOut, tlistOut, dlistOut, wlistOutL, slistOutL, clistOutL, tlistOutL, dlistOutL, wlistOutR, slistOutR, clistOutR, tlistOutR, dlistOutR


def convert2SenLv(llist, slist, wlist, lists):
    llistOut = []
    wlistOut = []
    listsOut = []
    
    for i in range(len(lists)):
        listsOut.append([])

    for idoc, doc in enumerate(slist):
        begin = -1
        end = -1
        thiswlist = []
        thissublists = []
        for i in range(len(lists)):
            thissublists.append([])
        thisllist = []
        for itok, token in enumerate(doc):
            if token[-1] == 'B':    # begin of a sentence
                begin = itok
            elif token[-1] == 'E':
                end = itok
                
                sen = flatten2Dlist(wlist[idoc][begin:end+1])
                thiswlist.append(sen)
                for i in range(len(lists)):
                    thissublists[i].append(lists[i][idoc][begin:end+1])
                
                isArg = False
                for i in range(begin,end+1):
                    if llist[idoc][i][-1] == 'B':
                         isArg = True
                         break
                
                if isArg:
                    thisllist.append('T')
                else:
                    thisllist.append('F')
        wlistOut.append(thiswlist)
        llistOut.append(thisllist)
        for i in range(len(lists)):
            listsOut[i].append(thissublists[i]) 
    
    return llistOut,wlistOut,listsOut

def loadListFull(dir,filelist,feature):
    llist = []
    wlist = []
    slist = []
    clist = []
    tlist = []
    dmlist = []
    pmlist = []
    d1list = []
    d2list = []
    
    for file in filelist:
        filename = dir + file
        if os.path.isfile(filename):
            with open(filename,'r') as f:
                lines = [line.rstrip('\n') for line in f]
                ll = []
                ww = []
                ss = []
                cc = []
                tt = []
                dmm = []
                pmm = [] 
                d11 = []
                d22 = []
                for line in lines:
                    try:
                        tokens = line.split('\t\t')
                        ll.append(tokens[0])
                        w = [tokens[1]]
                        ww.append(w)
                        c = tokens[2]
                        cc.append(c)
                        s = tokens[3]
                        ss.append(s)
                        t = tokens[4]
                        tt.append(t)
                        dm = tokens[5]
                        dmm.append(dm)
                        pm = tokens[6]
                        pmm.append(pm)
                        d1 = tokens[7]
                        d11.append(d1)
                        d2 = tokens[8]
                        d22.append(d2)
                    except:
                        print filename
                        continue
                    
                llist.append(ll)
                wlist.append(ww)
                slist.append(ss)
                clist.append(cc)
                tlist.append(tt)
                dmlist.append(dmm)
                pmlist.append(pmm)
                d1list.append(d11)
                d2list.append(d22)
                
    return llist, wlist, slist, clist, tlist, dmlist, pmlist, d1list,d2list

def flatten2Dlist(listlist):
    out = [item for sublist in listlist for item in sublist]
    
    return out

def getMaxDoc(wlists):
    dmax = 0
    fmax = 0
    for wlist in wlists:
        for words in wlist:
            for features in words:
                if len(features)>fmax:
                    fmax = len(features)
            if len(words)>dmax:
                dmax = len(words)    
    return fmax,dmax

def getMaxDoc1D(wlists):
    fmax = 0
    for wlist in wlists:
        for words in wlist:
            if len(words)>fmax:
                fmax = len(words) 
    return fmax

def dicLookUp(wlist,voc,samples,tLength,fLength):
    wArray = np.zeros((samples,tLength*fLength),dtype='int32')
    for idxw,w in enumerate(wlist):
        idx = 0
        for idxf,f in enumerate(w):
            for idxt,t in enumerate(f):                
                if t in voc:
                    wArray[idxw][idx] = voc[t]
                else:
                    wArray[idxw][idx] = 1  ## unknown word
                idx += 1
    return wArray

def dicLookUp2D(wlist,voc,samples,tLength):
    wArray = np.zeros((samples,tLength),dtype='int32')
    for idxw,w in enumerate(wlist):
        for idxt,t in enumerate(w):
            if t in voc:
                wArray[idxw][idxt] = voc[t]
            else:
                wArray[idxw][idxt] = 1  ## unknown word
    return wArray

def labelLookUp(llist,voc,samples,tLength,lSize):
    lArray = np.zeros((samples,tLength,lSize),dtype='int32')
    for il,l in enumerate(llist):
        for it,t in enumerate(l):
            lArray[il][it][voc[t]] = 1
    return lArray

def labelLookUp2D(llist,voc,samples,lSize):
    lArray = np.zeros((samples,lSize),dtype='int32')
    for il,l in enumerate(llist):
        lArray[il,voc[l]] = 1
    return lArray

def labelLookUpOri(llist,voc,samples,tLength,lSize):
    lArray = []
    for l in llist:
        llArray = [-1.]*tLength
        for it,t in enumerate(l):
            llArray[it] = voc[t]
        lArray.append(llArray)
    return np.array(lArray)

def dicLookUpFlat(llist,ldic):
    out = []
    for l in llist:
        out.append(ldic[l])
    return out

def predictFromClass(classes,ignores):
    for ignore in ignores:
        classes[:,:,ignore] = 0
    return np.argmax(classes,axis=-1)

def predictFromClass2D(classes,ignores):
    for ignore in ignores:
        classes[:,ignore] = 0
    return np.argmax(classes,axis=-1)

def predictFromClassNoPadding(classes,ignores):
    maxList = []
    for labellist in classes:
        maxVec = []
        for labels in labellist:
            for ignore in ignores:
                labels[ignore] = 0
            maxVec.append(np.argmax(labels))
        maxList.append(maxVec)

    return maxList

def filterDisLabel(classes,ignores):
    dis = []
    for i in range(classes.shape[0]):
        subdis = []
        for j in range(classes.shape[1]):
            subsubdis = []
            for k in range(classes.shape[2]):
                if k not in ignores:
                    subsubdis.append(classes[i,j,k])
            subdis.append(subsubdis)
        dis.append(subdis)
    return np.array(dis)

def filterDisLabel2D(classes,ignores):
    dis = []
    for i in range(classes.shape[0]):
        subdis = []
        for j in range(classes.shape[1]):
            if j not in ignores:
                subdis.append(classes[i,j])
        dis.append(subdis)
    return np.array(dis)

def findTrues(rlist,prediction,ignore):
    r = np.zeros((len(rlist),))
    for i in range(len(rlist)):
        if rlist[i] in ignore:
            if rlist[i] == prediction[i]:
                r[i] = 1
            else:
                r[i] = 0
    return r

def getSplitDataDir(dir,dev):
    seed = 9487
    random.seed(seed)
    trainFiles = os.listdir(dir + 'train/')
    testFiles = os.listdir(dir + 'test/')
      
    trainlist = []
    testlist = []
    
    for file in trainFiles:
        if file.endswith(".txt"):
            trainlist.append(file)
    
    for file in testFiles:
        if file.endswith(".txt"):
            testlist.append(file)
    
    trainList = []
    devList = []
    random.shuffle(trainlist)
    tSize = int(len(trainlist) * dev)
    
    for i,item in enumerate(trainlist):
        if i<tSize:
            devList.append(item)
        else:
            trainList.append(item)
    return trainList,devList,testlist

def getSplitDataFile(files,dev):
    seed = 9487
    random.seed(seed)
    
    with open(files[0],'r') as f0, open(files[1],'r') as f1:
        trainlist = [line.rstrip('\n') for line in f0]
        testlist = [line.rstrip('\n') for line in f1]
    
    trainList = []
    devList = []
    random.shuffle(trainlist)
    tSize = int(len(trainlist) * dev)
    
    for i,item in enumerate(trainlist):
        if i<tSize:
            devList.append(item)
        else:
            trainList.append(item)
    return trainList,devList,testlist
    
def nFolds(dir,folds):
    files = os.listdir(dir)
    
    superlist=[]
    for i in range(folds):
        alist = []
        superlist.append(alist)
           
    tSize = int(len(files)/folds)
    random.shuffle(files)
    for i,item in enumerate(files):
        for f in range(folds-1):
            if i >= (f)*tSize and i < (f+1)*tSize :
                superlist[f].append(item)
        if i >= (folds-1)*tSize:
            superlist[folds-1].append(item)
    return superlist    

def getFolds(alllist,fold,totalFolds):
    trainlist = []
    devlist = []
    testlist = []
    for i,l in enumerate(alllist):
        if i == fold:
            testlist = l
        elif i == (fold+1) % totalFolds:
            devlist = l
        else:
            trainlist.extend(l)
    return trainlist,devlist,testlist

def genLDic():
    lDic = {}
    lDic['<>'] = 0
    lDic['Arg-B'] = 1
    lDic['Arg-I'] = 2
    lDic['Arg-O'] = 3
    
    return lDic

def genInvLDic():
    lDic = {}
    lDic[0] = '<>'
    lDic[1] = 'Arg-B'
    lDic[2] = 'Arg-I'
    lDic[3] = 'Arg-O'
    
    return lDic

def genBLDic():
    lDic = {}
    lDic['<>'] = 0
    lDic['B'] = 1
    lDic['O'] = 2
    
    return lDic

def genELDic():
    lDic = {}
    lDic['<>'] = 0
    lDic['E'] = 1
    lDic['O'] = 2
    
    return lDic

def genLDicSenLv():
    lDic = {}
    lDic['<>'] = 0
    lDic['T'] = 1
    lDic['F'] = 2
    
    return lDic

def genBinaryLDic():
    lDic = {}
    lDic['<>'] = 0
    lDic['Arg-I'] = 1
    lDic['Arg-O'] = 2
    
    return lDic

def genSCTDic(blist):
    bDic = {}
    allB = []
    for b in blist:
        allB.extend(b)
        
    bset=list(set(allB))
    bDic['<zero>'] = 0
    bDic['<unk>'] = 1
    for i,b in enumerate(bset):
        bDic[b] = i+2
        
    return bDic

def filterPr(target,ref,ignore):
    out = []
    for ir,r in enumerate(ref):
        thisout=[]
        for irr,rr in enumerate(r):
            if rr not in ignore:
                thisout.append(target[ir,irr,1:])
        out.append(thisout)
    return out

def filterLabel(target,ref,ignore,keepShape=False):
    out = []
    for ir,r in enumerate(ref):
        if keepShape:
            thisout=[]
        for irr,rr in enumerate(r):
            if rr not in ignore:
                if keepShape:
                    thisout.append(target[ir][irr])
                else:
                    out.append(target[ir][irr])
        if keepShape:
            out.append(thisout)
    return out

def filterLabel2D(target,ref,ignore):
    out = []
    for ir,r in enumerate(ref):
        if r not in ignore:
            out.append(target[ir])
    return out

def filterLabelNoPadding(target,ref,ignore):
    out = []
    for ir,r in enumerate(ref):
        for irr,rr in enumerate(r):
            if rr not in ignore:
                out.append(target[ir][irr])
    return out

def filterDisPadding(target,ref,ignore):
    out = []
    for ir,r in enumerate(ref):
        for irr,rr in enumerate(r):
            if rr not in ignore:
                out.append(target[ir,irr,:])
    return normalize(np.array(out),'l1')

def filterDisPadding2D(target,ref,ignore):
    out = []
    for ir,r in enumerate(ref):
        if r not in ignore:
            out.append(target[ir,:])
    return normalize(np.array(out),'l1')


def genInputSCT(blist,bDic,samples,maxLength):
    bArray = np.zeros((samples,maxLength),dtype='int32')
    for idxb,b in enumerate(blist):
        for ibb,bb in enumerate(b):
            if bb in bDic:
                bArray[idxb][ibb] = bDic[bb]
            else:
                bArray[idxb][ibb] = 1  ## unknown word
    return bArray

def genHiddenInputSCT(blist,bDic,samples,maxLength):
    bArray = np.zeros((samples,maxLength,len(bDic)),dtype='int32')
    for idxb,b in enumerate(blist):
        for ibb,bb in enumerate(b):
            if bb in bDic:
                bArray[idxb][ibb][bDic[bb]] = 1
            else:
                bArray[idxb][ibb][1] = 1  ## unknown word
    return bArray

def convert2BIE(flist):
    flistout = []
    for f in flist:
        fout = []
        
        for iff,ff in enumerate(f):
            ffout = ff
            if ff[-1] != 'B':
                if iff+1 == len(f):
                    ffout = ffout[:-1] + 'E'
                elif f[iff+1][-1] == 'B':
                    ffout = ffout[:-1] + 'E'
                else:
                    ffout = ffout[:-1] + 'I'
            fout.append(ffout)
        flistout.append(fout)
    
    return flistout

def fineGrainDM(dmlist,wlist):
    finedmlist = []
    for idm,dm in enumerate(dmlist):
        finedm = []
        dms = []
        finedmm = ''
        for idmm,dmm in enumerate(dm):
            if dmm[-1] == 'B':
                if len(finedmm) != 0:
                    dms.append(finedmm)
                finedmm =  wlist[idm][idmm][0].lower()
            elif dmm[-1] == 'I':
                finedmm +=  ' '+wlist[idm][idmm][0].lower()
            elif dmm[-1] == 'E':
                finedmm +=  ' '+wlist[idm][idmm][0].lower()
                dms.append(finedmm)
                finedmm = ''
            else:
                if len(finedmm) != 0:
                    dms.append(finedmm)
        dmidx = -1
        for dmm in dm:
            if dmm[-1] == 'B' or dmm[-1] == 'I' or dmm[-1] == 'E':
                if dmm[-1] == 'B':
                    dmidx += 1
                finedmm = dms[dmidx] + '-' + dmm[-1]
            else:
                finedmm = dmm
            finedm.append(finedmm)
        finedmlist.append(finedm)
        
    return finedmlist

def locateDM(dmlist,wlist,slist,trainlist=None):
    finedmlist = []
    for idm,dm in enumerate(dmlist):
        finedm = []
        dms = []
        finedmm = ''
        hasdmlist = []
        for idmm,dmm in enumerate(dm):
            s = slist[idm][idmm][-1]
            if s == 'B':
                hasDM = False
                if idmm == len(dm)-1:
                    hasdmlist.append(hasDM)
                elif slist[idm][idmm+1][-1] == 'B':
                    hasdmlist.append(hasDM)
            else:
                if idmm == len(dm)-1:
                    hasdmlist.append(hasDM)
                elif slist[idm][idmm+1][-1] == 'B':
                    hasdmlist.append(hasDM)
            if dmm[-1] == 'B':
                if len(finedmm) != 0 or idmm == len(dm)-1:
                    dms.append(finedmm)
                    finedmm = ''
                finedmm =  wlist[idm][idmm][0].lower()
                hasDM = True
            elif dmm[-1] == 'I':
                finedmm +=  ' '+wlist[idm][idmm][0].lower()
            elif dmm[-1] == 'E':
                finedmm +=  ' '+wlist[idm][idmm][0].lower()
                dms.append(finedmm)
                finedmm = ''
            else:
                if len(finedmm) != 0:
                    dms.append(finedmm)
                    finedmm = ''

        dmidx = -1
        hasDMidx = -1
        status = 'B'
        for idmm,dmm in enumerate(dm):
            s = slist[idm][idmm][-1]           
            if s == 'B':
                hasDMidx += 1
                status = 'B'
                if dmm[-1] == 'B' or dmm[-1] == 'I' or dmm[-1] == 'E':
                    if dmm[-1] == 'B':
                        dmidx += 1
                    locateDM = dms[dmidx]
                    status = 'A'
                else:
                    status = 'B'
                    try:
                        if hasdmlist[hasDMidx]:
                            locateDM = dms[dmidx+1] + '-' + status
                        else:
                            locateDM = 'O'
                    except:
                        print 'hasDMidx',hasDMidx
                        print 'id',trainlist[idm]

            else:
                if dmm[-1] == 'B' or dmm[-1] == 'I' or dmm[-1] == 'E':
                    if dmm[-1] == 'B':
                        dmidx += 1
                        
                    try:
                        locateDM = dms[dmidx]
                    except:
                        print 'hasDMidx',dmidx
                        print 'hasDMidx',len(dms)
                        print 'id',trainlist[idm]
                        print 'dms',dms
                    status = 'A'
                else:
                    if hasdmlist[hasDMidx]:
                        if status == 'B':
                            locateDM = dms[dmidx+1] + '-' + status
                        else:
                            locateDM = dms[dmidx] + '-' + status
                    else:
                        locateDM = 'O'
            finedm.append(locateDM)
        finedmlist.append(finedm)
        
    return finedmlist

def locateFineDM(dmlist,wlist,slist):
    finedmlist = []
    for idm,dm in enumerate(dmlist):
        finedm = []
        dms = []
        finedmm = ''
        hasdmlist = []
        for idmm,dmm in enumerate(dm):
            s = slist[idm][idmm][-1]
            if s == 'B':
                hasDM = False
            else:
                if idmm == len(dm)-1:
                    hasdmlist.append(hasDM)
                elif slist[idm][idmm+1][-1] == 'B':
                    hasdmlist.append(hasDM)
            if dmm[-1] == 'B':
                if len(finedmm) != 0:
                    dms.append(finedmm)
                    finedmm = ''
                finedmm =  wlist[idm][idmm][0].lower()
                hasDM = True
            elif dmm[-1] == 'I':
                finedmm +=  ' '+wlist[idm][idmm][0].lower()
            elif dmm[-1] == 'E':
                finedmm +=  ' '+wlist[idm][idmm][0].lower()
                dms.append(finedmm)
                finedmm = ''
            else:
                if len(finedmm) != 0:
                    dms.append(finedmm)
                    finedmm = ''

        dmidx = -1
        hasDMidx = -1
        status = 'Before'
        for idmm,dmm in enumerate(dm):
            s = slist[idm][idmm][-1]           
            if s == 'B':
                hasDMidx += 1
                status = 'Before'
                if dmm[-1] == 'B' or dmm[-1] == 'I' or dmm[-1] == 'E':
                    if dmm[-1] == 'B':
                        dmidx += 1
                    locateDM = dms[dmidx] + '-' + dmm[-1]
                    status = 'After'
                else:
                    status = 'Before'
                    if hasdmlist[hasDMidx]:
                        locateDM = dms[dmidx+1] + '-' + status
                    else:
                        locateDM = 'O'
            else:
                if dmm[-1] == 'B' or dmm[-1] == 'I' or dmm[-1] == 'E':
                    if dmm[-1] == 'B':
                        dmidx += 1
                    locateDM = dms[dmidx] + '-' + dmm[-1]
                    status = 'After'
                else:
                    if hasdmlist[hasDMidx]:
                        if status == 'Before':
                            locateDM = dms[dmidx+1] + '-' + status
                        else:
                            locateDM = dms[dmidx] + '-' + status
                    else:
                        locateDM = 'O'
            finedm.append(locateDM)
        finedmlist.append(finedm)
        
    return finedmlist

def categorical_crossentropy_ignore(output, target, from_logits=False):
    ignore = 0
    
    if from_logits:
        output = T.nnet.softmax(output)
    else:
        output /= output.sum(axis=-1, keepdims=True)
    output = T.clip(output, _EPSILON, 1.0 - _EPSILON)
    
    filter1 = K.ones_like((K.shape(output)[:-1],1))
    filter2 = K.zeros_like((K.shape(output)[:-1],K.shape(output)[-1]-1))
    
    print K.shape(output)
    print K.shape(target)
    sys.exit()
    
    return T.nnet.categorical_crossentropy(output, target)


def dicLookUpNoPadding(wlist,voc,samples,tLength,fLength):
    wArray = []
    
    for w in wlist:
        wVec = []
        for f in w:
            for t in f:
                if t in voc:
                    wVec.append(voc[t])
                else:
                    wVec.append(1)
        wArray.append(wVec)
            
    return wArray

def genInputSCTNoPadding(blist,bDic,samples,maxLength):
    bArray = []
    for b in blist:
        bVec = []
        for bb in b:
            if bb in bDic:
                bVec.append(bDic[bb])
            else:
                bVec.append(1)  ## unknown word
        bArray.append(bVec)
    return bArray

def labelLookUpNoPadding(llist,voc,samples,tLength,lSize):
    lArray = []
    for l in llist:
        lVec = []
        for t in l:
            lvec = np.zeros((4,))
            lvec[voc[t]] = 1
        lVec.append(lvec)
    lArray.append(lVec)
    return lArray

def BIOconflict(tags):
    # B,I,O: 1,2,3
    
    status = ''
    conflict = 0
    for tag in tags:
        if tag == 2 and status == 3:
            conflict += 1
        status = tag
    return conflict

def BIO2IO(llist):
    out = []
    
    for l in llist:
        thisout = []
        for ll in l:
            if ll[-1] == 'B':
                thisout.append(ll[:-1] + 'I')
            else:
                thisout.append(ll)
        out.append(thisout)
    
    return out

def IO2BIO(pred):
    out = []
    
    for idx in range(len(pred)):
        if idx == 0:   # first token
            if pred[idx] == 1: # I
                out.append(1)   # B
            else:   # O
                out.append(3)   # O
        elif pred[idx-1] == 2 and pred[idx] == 1: # OI
            out.append(1)   # B
        else:
            out.append(pred[idx]+1)
    
    return out

def balanceSampling(clists,llist,method='over'):
    unq, unq_idx = np.unique(llist,return_inverse=True)
    unq_cnt = np.bincount(unq_idx)
    if method == 'over':
        cnt = np.max(unq_cnt)
    elif method == 'down':
        cnt = np.min(unq_cnt)
    
    outClists = []
    for clist in clists:
        outClist = np.empty((cnt*len(unq),) + clist.shape[1:], clist.dtype)
        outClists.append(outClist)
    
    outLlist = []
    
    for j in xrange(len(unq)):
        indices = np.random.choice(np.where(unq_idx==j)[0],cnt)
        for ic,clist in enumerate(clists):
            outClists[ic][j*cnt:(j+1)*cnt] = clist[indices]
        outLlist.extend([unq[j]]*cnt)
    return outClists,outLlist

def bestBIO(prediction,gold,slist):
    pred = []
    sid = -1
    for itok,tok in enumerate(slist):
        if tok[-1] == 'B':
            sid += 1
            begin = itok
        elif tok[-1] == 'E':
            end = itok
            
            if prediction[sid] == 1: #True
                pred.extend(gold[begin:end+1])
            else:
                pred.extend([3]*(end-begin+1))
    return pred

def printOutSenLabel(slists,llists,fnlists):
    
    with open('../../../data/sen_label.txt','w') as f:
        for ilist in range(len(fnlists)):
            sid = -1
            for idoc,fn in enumerate(fnlists[ilist]):
                labels = []               
                for itok,tok in enumerate(slists[ilist][idoc]):
                    if tok[-1] == 'B':
                        sid += 1
                    elif tok[-1] == 'E':
                        if llists[ilist][sid] == 1:
                            labels.append('T')
                        else:
                            labels.append('F')
                f.write(fn + '\t' + '\t'.join(labels) + '\n')
    return

def filterOut(senLabel,slist,llist,flists,type):
    flistsOut = []
    for i in range(len(flists)):
        flistsOut.append([])
    llistOut = []
    sid = -1
    for idoc,doc in enumerate(slist):
        
        subflistsOut = []
        subllistOut = []
        for i in range(len(flists)):
            subflistsOut.append([])
        for itok, tok in enumerate(doc):
            if tok[-1] == 'B':
                sid += 1
                begin = itok
            elif tok[-1] == 'E':
                end = itok
                
                if senLabel[sid] == 1:
                    for fidx,flist in enumerate(flists):
                        subflistsOut[fidx].extend(flist[idoc][begin:end+1])
                    
                    labels = llist[idoc][begin:end+1]
                    for iloc,loc in enumerate(labels):
                        if type == 'B':
                            if loc[-1] == 'B':
                                subllistOut.append('B')
                            else:
                                subllistOut.append('O')
                        elif type == 'E':
                            if iloc < len(labels)-1: # not last token
                                if labels[iloc][-1] == 'I' and labels[iloc+1][-1] == 'O':
                                    subllistOut.append('E')
                                else:
                                    subllistOut.append('O')
                            else:
                                if labels[iloc][-1] == 'I':
                                    subllistOut.append('E')
                                else:
                                    subllistOut.append('O')
        for i in range(len(flists)):
            flistsOut[i].append(subflistsOut[i])
        
        llistOut.append(subllistOut)
        
    return llistOut,flistsOut

def filterOutSenLv(senLabel,slist,llist,flists,type):
    flistsOut = []
    for i in range(len(flists)):
        flistsOut.append([])
    llistOut = []
    sid = -1
    for idoc,doc in enumerate(slist):
        for itok, tok in enumerate(doc):
            if tok[-1] == 'B':
                sid += 1
                begin = itok
                
                subflistsOut = []
                subllistOut = []
                for i in range(len(flists)):
                    subflistsOut.append([])
                
            elif tok[-1] == 'E':
                end = itok
                
                if senLabel[sid] == 1:
                    for fidx,flist in enumerate(flists):
                        subflistsOut[fidx].extend(flist[idoc][begin:end+1])
                    
                    labels = llist[idoc][begin:end+1]
                    for iloc,loc in enumerate(labels):
                        if type == 'B':
                            if loc[-1] == 'B':
                                subllistOut.append('B')
                            else:
                                subllistOut.append('O')
                        elif type == 'E':
                            if iloc < len(labels)-1: # not last token
                                if labels[iloc][-1] == 'I' and labels[iloc+1][-1] == 'O':
                                    subllistOut.append('E')
                                else:
                                    subllistOut.append('O')
                            else:
                                if labels[iloc][-1] == 'I':
                                    subllistOut.append('E')
                                else:
                                    subllistOut.append('O')

                    for i in range(len(flists)):
                        flistsOut[i].append(subflistsOut[i])
                
                    llistOut.append(subllistOut)
        
    return llistOut,flistsOut


def pipelinePred(slist,senLabel,bLabel,eLabel):
    pred = []
    
    sid = -1
    lidx = 0
    for itok,tok in enumerate(slist):
        if tok[-1] == 'B':
            sid += 1
            begin = itok
        elif tok[-1] == 'E':
            label = senLabel[sid]
            end = itok
            senLen = end - begin + 1
            
            if label == 1:  # argu unit
                bRegion = bLabel[lidx:lidx+senLen]
                eRegion = eLabel[lidx:lidx+senLen]
                lidx = lidx + senLen
                
                unit_b = -1
                unit_e = -1
                for ib,b in enumerate(bRegion): 
                    if b == 1: # begin
                        unit_b = ib
                        break
                for ie,e in reversed(list(enumerate(eRegion))): 
                    if e == 1: # begin
                        unit_e = ie
                        break
                
                if unit_b != -1 and unit_e != -1 and unit_e > unit_b:
                    pre = [3] * unit_b
                    
                    post = [3] * (senLen-(unit_e+1))
                    arg = [1] + [2]*(senLen-len(pre)-len(post)-1)
                    
                    pred.extend(pre + arg + post)
                else:
                    pred.extend([3]*senLen)
            else:
                pred.extend([3]*senLen)
    return pred

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def decode_batch(test_func,inputW,inputS,inputC,inputT,inputDM):
    out = test_func([inputW,inputS,inputC,inputT,inputDM])[0]
    
    print 'out:',out

    print inputW,inputS,inputC,inputT,inputDM
    raw_input("Press Enter to continue...")
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, :], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        ret.append(out_best)
        
    return ret

def sigTestStab(x,target=0.867):
    t,p = ttest_1samp(x, target)
    
    return t,p

def randomFolds(gold,pred,nFold):
    golds = []
    preds = []
    
    for i in range(nFold):
        aGold = []
        aPred = []
        golds.append(aGold)
        preds.append(aPred)
    
    seed = 9487
    random.seed(seed)
    
    combine = list(zip(gold,pred))
    random.shuffle(combine)
    gold[:],pred[:] = zip(*combine)

    tSize = int(len(gold) / nFold)
    
    for i,item in enumerate(gold):
        index = i / tSize
        golds[index].append(item)
        preds[index].append(pred[i])
        
    return golds,preds
    


