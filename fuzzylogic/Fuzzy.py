#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[3]:


class Input:
    def __init__(self, name, x, y, x0):
        self.name = name
        self.points = [(x[i], y[i]) for i in range(len(x))]
        self.mi = self.getMi(x0)
        
    def getY(self, x1, y1, x2, y2, x0):
        if y1 == y2:
            return y1
        if y1 < y2:
            return (x0-x1)/(x2-x1)
        
        return (x2-x0)/(x2-x1)
        
    def getMi(self, x0):
        #Finds between self.points is x0
        #And calculates mi for that point
        if x0 < self.points[0][0]:
            return self.points[0][1]
        if x0 > self.points[-1][0]:
            return self.points[-1][1]
        
        for i in range(1, len(self.points)):
            x1 = self.points[i-1][0]
            x2 = self.points[i][0]
            
            if x0 >= x1 and x0 < x2:
                y1 = self.points[i-1][1]
                y2 = self.points[i][1]
                return self.getY(x1, y1, x2, y2, x0)
        return -1
        


# In[4]:


class Output:
    #c is the average of x for points where y == 1
    #mi is calculated later, based on the rules
    def __init__(self, name, x, y):
        self.name = name
        sumX = 0
        n = 0
        self.points = []
        for i in range(len(x)):
            self.points.append((x[i], y[i]))
            if y[i] == 1:
                sumX += x[i]
                n += 1
                
        self.mi = 0
        self.value = sumX/n


# In[5]:


#define enum for logical values 
from enum import Enum, unique
@unique
class Logic(Enum):
    OR = 0
    AND = 1
    
#Definicija operacija u fazi logici
class Rule:
    def __init__(self, mfi1, mfi2, mfo, logic):
        self.mfInput1 = mfi1
        self.mfInput2 = mfi2
        self.mfOutput = mfo
        if logic == Logic.OR:
            self.mfOutput.mi = max(self.mfOutput.mi, max(self.mfInput1.mi, self.mfInput2.mi))
        else:
            self.mfOutput.mi = max(self.mfOutput.mi, min(self.mfInput1.mi, self.mfInput2.mi))


# <h3> Intersection fuzzy logic model </h3>
# <img src="https://www.fhwa.dot.gov/publications/research/safety/05078/images/fig2.gif">

# In[120]:


def func(q, p):
    
    #Racuna se za svaki pravac odvojeno
    
    #q - amount of cars in two lanes on one side
    #p - amount of cars in two lanes on the opposing side
    
    #Automobili sa jedne strane (dve trake sabrano)
    amount1 = []
    amount1.append(Input("malo", [5, 15], [1, 0], q))
    amount1.append(Input("srednje", [10, 20, 30, 35], [0, 1, 1, 0], q))
    amount1.append(Input("puno", [33, 100], [0, 1], q))
    
     #Automobili sa druge strane (dve trake sabrano)
    amount2 = []
    amount2.append(Input("malo", [5, 15], [1, 0], p))
    amount2.append(Input("srednje",  [10, 20, 30, 35], [0, 1, 1, 0], p))
    amount2.append(Input("puno", [33, 100], [0, 1], p))
    
    
    #Duzina tranjanja semafora
    length = []
    length.append(Output("kratko", [25, 35], [1, 0]))
    length.append(Output("srednje", [30, 55, 65, 75], [0, 1, 1, 0]))
    length.append(Output("dugo", [65, 100], [0, 1]))  
    
    rules = []
    #Ima malo automobila sa obe strane => kratko traje zeleno svetlo
    rules.append(Rule(amount1[0], amount2[0], length[0], Logic.AND))
    
    #Na jednoj strani ima malo automobila a na drugoj srednja kolicna automobila
    # => srednje trajanje zelenog svetla
    rules.append(Rule(amount1[0], amount2[1], length[1], Logic.AND))
    rules.append(Rule(amount1[1], amount2[0], length[1], Logic.AND))
    #Srednja kolicina automobila na obe strane => srednje trajanje zelenog svetla
    rules.append(Rule(amount1[1], amount2[1], length[1], Logic.AND))
    #Malo automobila na jednoj strani i puno na drugoj => srednje trajanje zelenog svetla
    rules.append(Rule(amount1[0], amount2[2], length[1], Logic.AND))
    rules.append(Rule(amount1[2], amount2[0], length[1], Logic.AND))
    #srednje i puno => dugo zeleno svetlo
    rules.append(Rule(amount1[1], amount2[2], length[2], Logic.AND))
    rules.append(Rule(amount1[2], amount2[1], length[2], Logic.AND))
    #puno i puno => dugo zeleno svetlo
    rules.append(Rule(amount1[2], amount2[2], length[2], Logic.AND))
    
    
    brojilac = 0
    imenioc = 0
    for o in length:
        brojilac += o.mi*o.value
        imenioc += o.mi
    solution = brojilac/imenioc
    
    print(solution)    


# In[121]:


func(12, 12)


# In[ ]:





# In[ ]:




