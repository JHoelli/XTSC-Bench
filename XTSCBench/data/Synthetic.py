import numpy as np 
from XTSCBench.metrics.synthetic_metrics import generateNewSample
import os

class Synthetic():
    """
    Enables the generation of Synthtic data in correspondance to [1].
    The code is largly inspired by https://github.com/ayaabdelsalam91/TS-Interpretability-Benchmark.

    References
    ----------
     [1] Ismail, Aya Abdelsalam, et al. "Benchmarking deep learning interpretability in time series predictions."
     Advances in neural information processing systems 33 (2020): 6441-6452.
    ----------
    """

    def __init__(self,NumTrainingSamples,NumTestingSamples, NumFeatures, NumTimeSteps, datasetsTypes= ["Middle", "SmallMiddle", "Moving_Middle", "Moving_SmallMiddle", "RareTime", "Moving_RareTime", "RareFeature","Moving_RareFeature","PostionalTime", "PostionalFeature"],dataGenerationTypes=[None ,"Harmonic", "PseudoPeriodic", "AutoRegressive" ,"CAR","NARMA" ], 
                 impTimeSteps=[30,14,30,15,6,6, 40,40,20,20],impFeatures=[1,1,1,1,1,1,1,1,1,1],
                 startImpTimeSteps=[10,18,10,18,22,22,5,5,None,None ], startImpFeatures=[0,0,0,0,0,0,0,0,0,0],
                 loc1=[None,None,None,None,None,None,None,None,1,1],loc2=[None,None,None,None,None,None,None,None,29,29],
                 freezeType = [None,None,None,None,None,None,None,None,"Feature","Time"], isMoving=[False,False,True,True,False,True,False,True,None,None],
                isPositional=[False,False,False,False,False,False,False,False,True,True],
                 models= None) -> None:
        '''
        Arguments:
            NumTrainingSamples int: Number of Training Example.
            NumTestingSamples int: Number of Testing Samles.
            NumFeatures int: Number of desired Features.
            NumTimeSteps int: Number if desired Time Tesos
            datasetsTypes array: Type of informative Feature. e.g., ["Middle", "SmallMiddle", "Moving_Middle", "Moving_SmallMiddle", "RareTime", "Moving_RareTime", "RareFeature","Moving_RareFeature","PostionalTime", "PostionalFeature"],dataGenerationTypes=[None ,"Harmonic", "PseudoPeriodic", "AutoRegressive" ,"CAR","NARMA" ], 
            impTimeSteps array: Time Steps to impute e.g., [30,14,30,15,6,6, 40,40,20,20]
            impFeatures array, Features to impute , e.g., [1,1,1,1,1,1,1,1,1,1]
            startImpTimeSteps array: Start Time Steps, e.g., [10,18,10,18,22,22,5,5,None,None ]
            startImpFeatures array: Start Feat ,e.g., [0,0,0,0,0,0,0,0,0,0],
            models array: This is still TODO None
        '''
        self.DatasetsTypes= datasetsTypes
        self.ImpTimeSteps=impTimeSteps
        self.ImpFeatures=impFeatures
        self.NumFeatures=NumFeatures
        self.NumTimeSteps=NumTimeSteps
        self.StartImpTimeSteps=startImpTimeSteps
        self.StartImpFeatures=startImpFeatures
        self.NumTrainingSamples=NumTrainingSamples
        self.NumTestingSamples=NumTestingSamples
        self.Loc1=loc1
        self.Loc2=loc2
        self.FreezeType = freezeType
        self.isMoving=isMoving
        self.isPositional=isPositional
        self.device = 'cpu'
        self.models=models
        self.DataGenerationTypes=dataGenerationTypes
        lengths=[len(startImpFeatures),len(startImpTimeSteps),len(isMoving),len(isPositional),len(impTimeSteps), len(impFeatures), len(loc1),len(loc2), len(datasetsTypes)]
        if not np.all(lengths) == len(datasetsTypes):
            raise Exception("All arrays speciyfying the dataset Types need to have the same Length!") 






   
    def createDatasets(self, datadir):
         self.data_dir=datadir
         for i in range(len(self.DatasetsTypes)):
            instance=Object()

            instance.ImpTimeSteps=self.ImpTimeSteps[i]
            instance.ImpFeatures=self.ImpFeatures[i]

            instance.StartImpTimeSteps=self.StartImpTimeSteps[i]
            instance.StartImpFeatures=self.StartImpFeatures[i]

            instance.Loc1=self.Loc1[i]
            instance.Loc2=self.Loc2[i]

            instance.FreezeType=self.FreezeType[i]
            instance.isMoving=self.isMoving[i]
            instance.isPositional=self.isPositional[i]


            for j in range(len(self.DataGenerationTypes)):
                if(self.DataGenerationTypes[j]==None):
                    instance.DataName=self.DatasetsTypes[i]+"_Box"
                else:
                   instance. DataName=self.DatasetsTypes[i]+"_"+self.DataGenerationTypes[j]
                instance.dataGenerationProcess=self.DataGenerationTypes[j]

                self.createSimulationDataProcesses(instance)

    def createSimulationDataProcesses(self, instance):



        if(instance.isPositional):
            print("Creating Positional Training Dataset" , instance.DataName)
            TrainingDataset  , TrainingDataset_MetaData= self.createPositionalDataset( self.NumTrainingSamples, instance)
            print("Creating Positional Testing Dataset", instance.DataName)
            TestingDataset ,TestingDataset_MetaData= self.createPositionalDataset(self.NumTestingSamples, instance)

        else:

            print("Creating Training Dataset", instance.DataName)
            TrainingDataset  , TrainingDataset_MetaData= self.createDataset( self.NumTrainingSamples, instance)
            print("Creating Testing Dataset", instance.DataName)
            TestingDataset ,TestingDataset_MetaData= self.createDataset(self.NumTestingSamples, instance)
            
        negIndex=-1
        posIndex=-1


        for i in range(TrainingDataset_MetaData.shape[0]):
                if(TrainingDataset_MetaData[i,0]==1):
                    posIndex=i
                else:
                    negIndex=i

                if(negIndex!=-1 and posIndex!=-1):
                    break



        if self.data_dir is not None:
            print("Saving Datasets...")
            if not os.path.isdir(self.data_dir+"Training"):
                os.mkdir(self.data_dir+"Training")
            if not os.path.isdir(self.data_dir+"Training/data"):
                os.mkdir(self.data_dir+"Training/data")
            if not os.path.isdir(self.data_dir+"Training/meta"):
                os.mkdir(self.data_dir+"Training/meta")
            if not os.path.isdir(self.data_dir+"Testing"):
                os.mkdir(self.data_dir+"Testing")
            if not os.path.isdir(self.data_dir+"Testing/data"):
                os.mkdir(self.data_dir+"Testing/data")
            if not os.path.isdir(self.data_dir+"Testing/meta"):
                os.mkdir(self.data_dir+"Testing/meta")
            np.save(self.data_dir+"Training/data/SimulatedTrainingData_"+ instance.DataName+"_F_"+str(self.NumFeatures)+"_TS_"+str(self.NumTimeSteps),TrainingDataset)
            np.save(self.data_dir+"Training/meta/SimulatedTrainingMetaData_"+instance.DataName+"_F_"+str(self.NumFeatures)+"_TS_"+str(self.NumTimeSteps),TrainingDataset_MetaData)

            np.save(self.data_dir+"Testing/data/SimulatedTestingData_"+instance.DataName+"_F_"+str(self.NumFeatures)+"_TS_"+str(self.NumTimeSteps),TestingDataset)
            np.save(self.data_dir+"Testing/meta/SimulatedTestingMetaData_"+instance.DataName+"_F_"+str(self.NumFeatures)+"_TS_"+str(self.NumTimeSteps),TestingDataset_MetaData)
    
    def createSample(self, instance,Target,start_ImpTS,end_ImpTS,start_ImpFeat,end_ImpFeat):
        sample=generateNewSample(instance.dataGenerationProcess, sampler="irregular", NumTimeSteps=self.NumTimeSteps, NumFeatures=self.NumFeatures)
        sample[start_ImpTS:end_ImpTS,start_ImpFeat:end_ImpFeat]=sample[start_ImpTS:end_ImpTS,start_ImpFeat:end_ImpFeat]+Target
        return sample
    
    def createPositionalDataset(self,NumberOFsamples, instance):
        DataSet = np.zeros((NumberOFsamples ,self.NumTimeSteps , self.NumFeatures  ))
        metaData= np.zeros((NumberOFsamples,5))
        Targets = np.random.randint(-1, 1,NumberOFsamples)

        TargetTS_Ends=np.zeros((NumberOFsamples,))
        TargetFeat_Ends=np.zeros((NumberOFsamples,))

        if (instance.FreezeType=="Feature"):

            TargetTS_Starts = np.random.randint(self.NumTimeSteps-instance.ImpTimeSteps, size=NumberOFsamples)		
            TargetFeat_Starts=np.zeros((NumberOFsamples,))

            for i in range (NumberOFsamples):
                if(Targets[i]==0):
                    Targets[i]=1
                    TargetYStart,TargetXStart = TargetTS_Starts[i], instance.Loc1
                else:
                    TargetYStart,TargetXStart = TargetTS_Starts[i], instance.Loc2

            # print(TargetXStart)
                TargetFeat_Starts[i]=TargetXStart

                TargetYEnd,TargetXEnd = TargetYStart+instance.ImpTimeSteps, TargetXStart+instance.ImpFeatures
                sample =self.createSample(instance,1,TargetYStart,TargetYEnd,TargetXStart,TargetXEnd)
                if(Targets[i]==-1):
                    Targets[i]=0

                TargetTS_Ends[i] = TargetTS_Starts[i]+instance.ImpTimeSteps
                TargetFeat_Ends[i] = TargetFeat_Starts[i]+instance.ImpFeatures

                DataSet[i,:,:,]=sample

        else:
            if self.NumFeatures >1:
                TargetFeat_Starts = np.random.randint( self.NumFeatures -instance.ImpFeatures, size=NumberOFsamples)
            else:
                TargetFeat_Starts = np.repeat(1, NumberOFsamples) 
            TargetTS_Starts=np.zeros((NumberOFsamples,))

            for i in range (NumberOFsamples):
                if(Targets[i]==0):
                    Targets[i]=1
                    TargetYStart,TargetXStart = instance.Loc1, TargetFeat_Starts[i]
                else:
                    TargetYStart,TargetXStart = instance.Loc2, TargetFeat_Starts[i]

                TargetTS_Starts[i]=TargetYStart

                TargetYEnd,TargetXEnd = TargetYStart+instance.ImpTimeSteps, TargetXStart+instance.ImpFeatures
                #TODO create Sample from Synthtic data
                sample = self.createSample(instance,1,TargetYStart,TargetYEnd,TargetXStart,TargetXEnd)
                if(Targets[i]==-1):
                    Targets[i]=0

                TargetTS_Ends[i] = TargetTS_Starts[i]+instance.ImpTimeSteps
                TargetFeat_Ends[i] = TargetFeat_Starts[i]+instance.ImpFeatures



                DataSet[i,:,:,]=sample


        #Label
        metaData[:,0]=Targets
        #Start important time
        metaData[:,1]=TargetTS_Starts
        #End important time
        metaData[:,2]=TargetTS_Ends
        #Start important feature
        metaData[:,3]=TargetFeat_Starts
        #End important feature
        metaData[:,4]=TargetFeat_Ends

        return DataSet, metaData

    def createDataset(self,NumberOFsamples, instance):

        DataSet = np.zeros((NumberOFsamples ,self.NumTimeSteps , self.NumFeatures))
        metaData= np.zeros((NumberOFsamples,5))
        Targets = np.random.randint(-1, 1,NumberOFsamples)

        TargetTS_Ends=np.zeros((NumberOFsamples,))
        TargetFeat_Ends=np.zeros((NumberOFsamples,))

        if(self.isMoving):
            TargetTS_Starts = np.random.randint(self.NumTimeSteps-instance.ImpTimeSteps, size=NumberOFsamples)

            if self.NumFeatures >1:
                TargetFeat_Starts = np.random.randint(self.NumFeatures-instance.ImpFeatures, size=NumberOFsamples)
            else:
                TargetFeat_Starts=np.repeat(1, NumberOFsamples)


        else:
            TargetTS_Starts=np.zeros((NumberOFsamples,))
            TargetFeat_Starts=np.zeros((NumberOFsamples,))

            TargetTS_Starts[:]= instance.StartImpTimeSteps 
            TargetFeat_Starts[:]= instance.StartImpFeatures


        for i in range (NumberOFsamples):
            if(Targets[i]==0):
                Targets[i]=1

            TargetTS_Ends[i],TargetFeat_Ends[i] = TargetTS_Starts[i]+instance.ImpTimeSteps, TargetFeat_Starts[i]+instance.ImpFeatures
            sample = self.createSample(instance,Targets[i],int(TargetTS_Starts[i]),int(TargetTS_Ends[i]),int(TargetFeat_Starts[i]),int(TargetFeat_Ends[i]))

            if(Targets[i]==-1):
                Targets[i]=0

            DataSet[i,:,:,]=sample

        #Label
        metaData[:,0]=Targets
        #Start important time
        metaData[:,1]=TargetTS_Starts
        #End important time
        metaData[:,2]=TargetTS_Ends
        #Start important feature
        metaData[:,3]=TargetFeat_Starts
        #End important feature
        metaData[:,4]=TargetFeat_Ends



        return DataSet , metaData

class Object(object):
    pass