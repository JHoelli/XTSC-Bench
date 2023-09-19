import numpy as np 

class Synthtic():
    def __init__(self,NumTrainingSamples,NumTestingSamples, NumFeatures, NumTimesteps, datasetsTypes= ["Middle", "SmallMiddle", "Moving_Middle", "Moving_SmallMiddle", "RareTime", "Moving_RareTime", "RareFeature","Moving_RareFeature","PostionalTime", "PostionalFeature"],dataGenerationTypes=[None ,"Harmonic", "PseudoPeriodic", "AutoRegressive" ,"CAR","NARMA" ], 
                 impTimeSteps=[30,14,30,15,6,6, 40,40,20,20],impFeatures=[1,1,1,1,1,1,1,1,1,1],
                 startImpTimeSteps=[10,18,10,18,22,22,5,5,None,None ], startImpFeatures=[0,0,0,0,0,0,0,0,0,0],
                 models= None) -> None:
        self.DatasetsTypes= datasetsTypes
        self.ImpTimeSteps=impTimeSteps
        self.ImpFeatures=impFeatures
        self.NumFeatures=NumFeatures
        self.NumTimesteps=NumTimesteps

        self.StartImpTimeSteps=startImpTimeSteps
        self.StartImpFeatures=startImpFeatures
        self.NumTrainingSamples=NumTrainingSamples
        self.NumTestingSamples=NumTestingSamples
        self.Loc1=[None,None,None,None,None,None,None,None,1,1]
        self.Loc2=[None,None,None,None,None,None,None,None,29,29]


        self.FreezeType = [None,None,None,None,None,None,None,None,"Feature","Time"]
        self.isMoving=[False,False,True,True,False,True,False,True,None,None]
        self.isPositional=[False,False,False,False,False,False,False,False,True,True]
        self.device = 'cpu'
        self.models=models
        self.DataGenerationTypes=dataGenerationTypes



   
    def createDatasets(self, datadir):
         self.data_dir=datadir
         for i in range(len(self.DatasetsTypes)):
            instance=dict()

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
                instance.DataGenerationProcess=self.DataGenerationTypes[j]

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
            #TODO Generate necessary folders
            print("Saving Datasets...")
            np.save(self.data_dir+"Training/data/SimulatedTrainingData_"+ instance.DataName+"_F_"+str(self.NumFeatures)+"_TS_"+str(self.NumTimeSteps),TrainingDataset)
            np.save(self.data_dir+"Training/meta/SimulatedTrainingMetaData_"+instance.DataName+"_F_"+str(self.NumFeatures)+"_TS_"+str(self.NumTimeSteps),TrainingDataset_MetaData)

            np.save(self.data_dir+"Testing/data/SimulatedTestingData_"+instance.DataName+"_F_"+str(self.NumFeatures)+"_TS_"+str(self.NumTimeSteps),TestingDataset)
            np.save(self.data_dir+"Testing/meta/SimulatedTestingMetaData_"+instance.DataName+"_F_"+str(self.NumFeatures)+"_TS_"+str(self.NumTimeSteps),TestingDataset_MetaData)
    
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
                #TODO create Sample from Synthtic data
                sample = createSample(instance,1,TargetYStart,TargetYEnd,TargetXStart,TargetXEnd)
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
                sample = createSample(args,1,TargetYStart,TargetYEnd,TargetXStart,TargetXEnd)
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
            sample = createSample(args,Targets[i],int(TargetTS_Starts[i]),int(TargetTS_Ends[i]),int(TargetFeat_Starts[i]),int(TargetFeat_Ends[i]))

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
