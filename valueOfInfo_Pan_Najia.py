

def expect(xDistribution, function):
    fxProduct=[px*function(x) for x, px in xDistribution.items()]
    expectation=sum(fxProduct)
    return expectation

def getUnnormalizedPosterior(prior, likelihood):

    #unnomarlizedPosterior={}
    #for S,Ps in prior. items():
     #   unnomarlizedPosterior[s]= prior[S]*likelihood[S]

    unnormalizedPosterior = {s: prior[s] * likelihood[s] for s in prior}
    return unnormalizedPosterior

def normalize(unnormalizedDistribution):

    getPe=sum(unnormalizedDistribution[s] for s in unnormalizedDistribution)
    normalizedPosterior = {s:unnormalizedDistribution[s]/getPe for s in unnormalizedDistribution}

    return normalizedPosterior

def getSumOfProbability(unnormalizedDistribution):

    getPe = sum(unnormalizedDistribution[s] for s in unnormalizedDistribution)
    
    return getPe

def getPosterior(prior, likelihood):

    unnormalizedDistribution=getUnnormalizedPosterior(prior,likelihood)
    posteriorDistribution=normalize(unnormalizedDistribution)

    return posteriorDistribution

def getMarginalOfData(prior, likelihood):

    unnormolizedDistribution= getUnnormalizedPosterior(prior,likelihood)
    getMarginalOfData=getSumOfProbability(unnormolizedDistribution)

    return getMarginalOfData

def getEU(action, sDistribution, rewardTable):
    
    utilityFunction= lambda s: rewardTable[s][action]
    getEU=expect(sDistribution,utilityFunction)

    return getEU

def getMaxEUFull(evidence, prior, likelihoodTable, rewardTable, actionSpace):

    if evidence is None:
        posteriorDistribution = prior
    else:
        posteriorDistribution = getPosterior(prior, likelihoodTable[evidence])

    EU={}
    for action in actionSpace:
        EU[action]= getEU(action,posteriorDistribution,rewardTable)

    maxEU = max(EU.values())
    return maxEU

def getValueOfInformationOfATest(evidenceSpace, getMarginalOfEvidence, getMaxEU):

    peDistribution={}
    for e in evidenceSpace:
        peDistribution[e]= getMarginalOfEvidence(e)

    valuesOfInformation=expect(peDistribution,getMaxEU)-getMaxEU(None)
    return valuesOfInformation



def main():
    
    prior={'Well 1 contains oil': 0.2, 'Well 2 contains oil': 0.4, 'Well 3 contains oil': 0.2, 'Well 4 contains oil': 0.2}
    
    actionSpace=['Buy Well 1', 'Buy Well 2', 'Buy Well 3', 'Buy Well 4']
    rewardTable={'Well 1 contains oil': {'Buy Well 1': 100, 'Buy Well 2': 0, 'Buy Well 3': 0, 'Buy Well 4': 0},
                 'Well 2 contains oil': {'Buy Well 1': 0, 'Buy Well 2': 100, 'Buy Well 3': 0, 'Buy Well 4': 0},
                 'Well 3 contains oil': {'Buy Well 1': 0, 'Buy Well 2': 0, 'Buy Well 3': 100, 'Buy Well 4': 0},
                 'Well 4 contains oil': {'Buy Well 1': 0, 'Buy Well 2': 0, 'Buy Well 3': 0, 'Buy Well 4': 100}}   
    
    testSpace=['Test Well 1', 'Test Well 2', 'Test Well 3', 'Test Well 4']
    evidenceSpace=['Microbe', 'No microbe']
    likelihoodTable={'Test Well 1':{'Microbe': {'Well 1 contains oil': 0.8, 'Well 2 contains oil': 0.1, 'Well 3 contains oil': 0.1, 'Well 4 contains oil': 0.1},
                                    'No microbe': {'Well 1 contains oil': 0.2, 'Well 2 contains oil': 0.9, 'Well 3 contains oil': 0.9, 'Well 4 contains oil': 0.9}},
                     'Test Well 2':{'Microbe': {'Well 1 contains oil': 0.1, 'Well 2 contains oil': 0.8, 'Well 3 contains oil': 0.1, 'Well 4 contains oil': 0.1},
                                    'No microbe': {'Well 1 contains oil': 0.9, 'Well 2 contains oil': 0.2, 'Well 3 contains oil': 0.9, 'Well 4 contains oil': 0.9}},
                     'Test Well 3':{'Microbe': {'Well 1 contains oil': 0.1, 'Well 2 contains oil': 0.1, 'Well 3 contains oil': 0.8, 'Well 4 contains oil': 0.1},
                                    'No microbe': {'Well 1 contains oil': 0.9, 'Well 2 contains oil': 0.9, 'Well 3 contains oil': 0.2, 'Well 4 contains oil': 0.9}},
                     'Test Well 4':{'Microbe': {'Well 1 contains oil': 0.1, 'Well 2 contains oil': 0.1, 'Well 3 contains oil': 0.1, 'Well 4 contains oil': 0.8},
                                    'No microbe': {'Well 1 contains oil': 0.9, 'Well 2 contains oil': 0.9, 'Well 3 contains oil': 0.9, 'Well 4 contains oil': 0.2}}}
    

    getMarginalOfEvidenceGivenTest=lambda test, evidence: getMarginalOfData(prior, likelihoodTable[test][evidence]) 
    
    getMaxEU=lambda test, evidence:getMaxEUFull(evidence, prior, likelihoodTable[test], rewardTable, actionSpace)
    
    getValueOfInformation=lambda test: getValueOfInformationOfATest(evidenceSpace,
                                                                    lambda evidence: getMarginalOfEvidenceGivenTest(test, evidence), 
                                                                    lambda evidence: getMaxEU(test, evidence))
    
    testExample1='Test Well 1'
    print(getValueOfInformation(testExample1))
    
    testExample2='Test Well 2'
    print(getValueOfInformation(testExample2))



    
if __name__=="__main__":
    main()      
