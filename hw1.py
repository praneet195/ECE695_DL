# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:39:35 2020

@author: singh671
"""
import random
import string

class People:
    def __init__(self,first_names,middle_names,last_names,order=None):
        self.first_names=first_names
        self.last_names=last_names
        self.middle_names=middle_names
        self.index=0
        self.order=order
        
    def __str__(self):
        return "First Names: "+str(self.first_names)+"\nMiddle Names: "+str(self.middle_names)+"\nLast Names: "+str(self.last_names) 
    
    def __call__(self):
        for i in sorted(self.last_names):
            print(i)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        result=""
        try:
            if self.order==None or self.order=="first_name_first":
                result = self.first_names[self.index]+" "+self.middle_names[self.index]+" "+self.last_names[self.index]
            elif self.order=="last_name_first":
                result = self.last_names[self.index]+" "+self.first_names[self.index]+" "+self.middle_names[self.index]
            elif self.order=="last_name_with_comma_first":
                result = self.last_names[self.index]+", "+self.first_names[self.index]+" "+self.middle_names[self.index]
        except IndexError:
            pass
        self.index += 1
        return result
    
  
class PeopleWithMoney(People):
    def __init__(self,first_names,middle_names,last_names,order=None):
        People.__init__(self,first_names,middle_names,last_names,order)
        self.wealth=[random.randint(0,1000) for i in range(10)]
        
    def __str__(self):
        return People.__str__(self)+"\nWealth : "+str(self.wealth)
        
    def __iter__(self):
        People.__iter__(self)
        return self
    
    def __next__(self):
        try:
            res=People.__next__(self)
            res += " "+ str(self.wealth[self.index-1])
        except IndexError:
            pass
        return res
    
    def __call__(self):
        People.__init__(self,self.first_names,self.middle_names,self.last_names,self.order)
        res={}
        for i in range(self.wealth.__len__()):
            res[self.wealth[i]]=str(People.__next__(self))
        res_sort=sorted(res.keys())
        for i in res_sort:
            print(res[i]+" "+str(i))
        
    
if __name__=="__main__":     
    
    random.seed(0)
    letters=string.ascii_lowercase
    lenght=5
    num=10
    first_names=[''.join(random.choice(letters) for i in range(lenght)) for i in  range(num)]
    middle_names=[''.join(random.choice(letters) for i in range(lenght)) for i in  range(num)]
    last_names=[''.join(random.choice(letters) for i in range(lenght)) for i in  range(num)]
    
    p1_obj=People(first_names,middle_names,last_names)
    #p1_obj.show()
    for i in range(num):
        print(next(p1_obj))
    print()
    
    p2_obj=People(first_names,middle_names,last_names,"last_name_first")
    #p2_obj.show()
    for i in range(num):
        print(next(p2_obj))
    print()
    
    p3_obj=People(first_names,middle_names,last_names,"last_name_with_comma_first")
    #p3_obj.show()
    for i in range(num):
        print(next(p3_obj))
    print()
    
    p3_obj()
    print()
    
    w_obj=PeopleWithMoney(first_names,middle_names,last_names,"first_name_first")
    for i in range(num):
        print(next(w_obj))
    print()
    
    w_obj()

