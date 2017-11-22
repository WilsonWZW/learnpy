# A dictionary about movie critics
critics = {'Lisa Rose': {'Lady in the Water':2.5, 'Snakes on a Plane':3.5, 'Just My Luck': 3.0, 'Superman Returns':3.5, 'You, Me and Dupree':2.5, 'The Night Listener': 3.0},
           'Gene Seymour': {'Lady in the Water':3.0, 'Snakes on a Plane':3.5, 'Just My Luck':1.5, 'Superman Returns':5.0, 'The Night Listener':3.0, 'You, Me and Dupree':3.5},
           'Michael Phillips':{'Lady in the Water': 2.5, 'Snakes on a Plane':3.0, 'Superman Returns':3.5, 'The Night Listener':4.0},
           'Claudia Puig': {'Snakes on a Plane':3.5, 'Just My Luck':3.0, 'The Night Listener':4.5, 'Superman Returns':4.0, 'You, Me and Dupree':2.5},
           'Mick LaSalle':{'Lady in the Water': 3.0, 'Snakes on a Plane':4.0, 'Just My Luck': 2.0, 'Superman Returns':3.0, 'The Night Listener':3.0, 'You, Me and Dupree':2.0},
           'Jack Matthews':{'Lady in the Water':3.0, 'Snakes on a Plane':4.0, 'The Night Listener':3.0, 'Superman Returns':5.0, 'You, Me and Dupree':3.5},
           'Toby':{'Snakes on a Plane':4.5, 'You, Me and Dupree':1.0, 'Superman Returns':4.0}}

# Similarity function
from math import sqrt

# Return a parameter about the similarity distance between person1 and person2
def sim_distance(prefs, person1, person2):
    # get list of shared_items
    si={}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1
    
    # if no similarity, return 0
    if len(si) == 0: return 0
    
    # sum of squares
    sum_of_squares = sum([pow(prefs[person1][item]-prefs[person2][item],2)
                          for item in prefs[person1] if item in prefs[person2]])
    return 1/(1+sqrt(sum_of_squares))

# Return Pearson correlation coefficient
def sim_pearson(prefs,p1,p2):
    # Get the list of items which were rated by both
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item] = 1
            
    # Get the number of items
    n = len(si)
    
    # if there is no similarity, Return 1
    if n==0: return 1
    
    # Sum all the preferences
    sum1 = sum([prefs[p1][it] for it in si])
    sum2 = sum([prefs[p2][it] for it in si])
    
    # Sum of squares
    sum1Sq = sum([pow(prefs[p1][it],2) for it in si])
    sum2Sq = sum([pow(prefs[p2][it],2) for it in si])
    
    # sum of multiplies
    pSum = sum([prefs[p1][it]*prefs[p2][it] for it in si])
    
    # Calculate Pearson correlation corefficient
    num = pSum - (sum1*sum2/n)
    den = sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
    if den==0: return 0
    
    r=num/den
    
    return r

# Get the top n matches
def topMatches(prefs, person, n=5, similarity=sim_pearson):
    scores = [(similarity(prefs, person, other),other) 
              for other in prefs if other!=person]
    
    # sort the scores
    scores.sort()
    scores.reverse()
    return scores[0:n]

# Use weighted average of others ratngs, to provide recommendations
def getRecommendations(prefs,person,similarity=sim_pearson):
    totals={}
    # Similarity sums
    simSums={}
    for other in prefs:
        # Do not compare with themselves
        if other == person: continue
        sim=similarity(prefs, person, other)
        
        # Ingore the sim<=0 scenarios.
        if sim <= 0: continue
        for item in prefs[other]:
            
            if item not in prefs[person] or prefs[person][item]==0:
                totals.setdefault(item,0)
                totals[item]+=prefs[other][item]*sim
                simSums.setdefault(item,0)
                simSums[item]+=sim
    
    # create a normalised list
    rankings = [(total/simSums[item],item) for item,total in totals.items()]
    
    rankings.sort()
    rankings.reverse()
    print totals.items()
    return rankings

# tranform person<->movie
def transformPrefs(prefs):
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
            result[item][person]=prefs[person][item]
    return result
            
            
            
            
        
    