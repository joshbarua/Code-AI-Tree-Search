from vllm import LLM, SamplingParams
import time

prompt = """You are an expert python programmer tasked with solving 3 problems. Wrap your solutions inside a code block.
PROBLEM 1: Write a python code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:                                                                                                      
QUESTION:                                                                                                                                                                                                                                                                       
You managed to send your friend to queue for tickets in your stead, but there is a catch: he will get there only if you tell him how much that is going to take. And everybody can only take one ticket at a time, then they go back in the last position of the queue if they need more (or go home if they are fine).
                                                                                                                                                                                                                                                                                
Each ticket takes one minutes to emit, the queue is well disciplined, [Brit-style](https://www.codewars.com/kata/english-beggars), and so it moves smoothly, with no waste of time.                                                                                             
                                                                                                                                                                                                                                                                                
You will be given an array/list/vector with all the people queuing and the *initial* position of your buddy, so for example, knowing that your friend is in the third position (that we will consider equal to the index, `2`: he is the guy that wants 3 tickets!) and the initial queue is `[2, 5, 3, 4, 6]`.
                                                                                                                                                                                                                                                                                
The first dude gets his ticket and the queue goes now like this `[5, 3, 4, 6, 1]`, then `[3, 4, 6, 1, 4]` and so on. In the end, our buddy will be queuing for 12 minutes, true story!                                                                                          
                                                                                                                                                                                                                                                                                
Build a function to compute it, resting assured that only positive integers are going to be there and you will be always given a valid index; but we also want to go to pretty popular events, so be ready for big queues with people getting plenty of tickets.                
                                                                                                                                                                                                                                                                                
[[hard core version](https://www.codewars.com/kata/queue-time-counter-hard-core-version/solutions/javascript) now available if you don't want the "easy" kata!]                                                                                                                 
def queue(queuers,pos):                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                
Use Call-Based format                                                                                                                                                                                                                                                           
ANSWER:                                                                                                                                                                                                                                                                         
```def queue(queuers,pos):                                                                                                                                                                                                                                                      
        return sum(min(queuer, queuers[pos] - (place > pos)) for place, queuer in enumerate(queuers))                                                                                                                                                                           
```                                                                                                                                                                                                                                                                             
PROBLEM 2: Write a python code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:                                                                                                      
QUESTION:                                                                                                                                                                                                                                                                       
Working from left-to-right if no digit is exceeded by the digit to its left it is called an increasing number; for example, 134468.                                                                                                                                             
                                                                                                                                                                                                                                                                                
Similarly if no digit is exceeded by the digit to its right it is called a decreasing number; for example, 66420.                                                                                                                                                               
                                                                                                                                                                                                                                                                                
We shall call a positive integer that is neither increasing nor decreasing a "bouncy" number; for example, 155349.                                                                                                                                                              
                                                                                                                                                                                                                                                                                
Clearly there cannot be any bouncy numbers below one-hundred, but just over half of the numbers below one-thousand (525) are bouncy. In fact, the least number for which the proportion of bouncy numbers first reaches 50% is 538.                                             
                                                                                                                                                                                                                                                                                
Surprisingly, bouncy numbers become more and more common and by the time we reach 21780 the proportion of bouncy numbers is equal to 90%.                                                                                                                                       
                                                                                                                                                                                                                                                                                
#### Your Task                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                
Complete the bouncyRatio function.                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                
The input will be the target ratio.                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                
The output should be the smallest number such that the proportion of bouncy numbers reaches the target ratio.                                                                                                                                                                   
                                                                                                                                                                                                                                                                                
You should throw an Error for a ratio less than 0% or greater than 99%.                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                
**Source**                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                
  - https://projecteuler.net/problem=112                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                
**Updates**                                                                                                                                                                                                                                                                     
                                                                    
  - 26/10/2015: Added a higher precision test case.
def bouncy_ratio(percent):                                                     
                                                                    
Use Call-Based format     
ANSWER:                                                                        
```def bouncy_ratio(percent):
        ans = 0
        for i in range(1, 10**10):
                s = str(i)
                res = ''.join(['=' if a == b else '>' if int(a) > int(b) else '<' for a, b in zip(s, s[1:])])                                                  
                ans += ('<' in res) and ('>' in res)
                if ans / i >= percent:                                                                                                  
                        return i                    
```                                   
PROBLEM 3:                      
QUESTION:
Generate and return **all** possible increasing arithmetic progressions of six primes `[a, b, c, d, e, f]` between the given limits. Note: the upper and lower limits are inclusive.                                                                                                                                          
                                                                    
An arithmetic progression is a sequence where the difference between consecutive numbers is the same, such as: 2, 4, 6, 8.                                                                                                                                                      

A prime number is a number that is divisible only by itself and 1 (e.g. 2, 3, 5, 7, 11)                                   

Your solutions should be returned as lists inside a list in ascending order of the first item (if there are multiple lists with same first item, return in ascending order for the second item etc) are the e.g: `[ [a, b, c, d, e, f], [g, h, i, j, k, l] ]` where `a < g`. If there are no solutions, return an empty list: 
`[]`                                                                           

## Examples                                                                    
def primes_a_p(lower_limit, upper_limit):                                      

Use Call-Based format                                                          
ANSWER:"""

prompts = [prompt for _ in range(500)]

llm = LLM(model='TheBloke/CodeLlama-7B-Python-AWQ', quantization='AWQ', dtype='float16')
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, use_beam_search=False, n=1, max_tokens=256)

start = time.time()
outputs = llm.generate(prompts, sampling_params)
print(time.time()-start)
