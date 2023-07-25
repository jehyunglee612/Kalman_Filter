#function that generate random list of random values between 10 and 18 
def generate_random_list():
    import random
    random_list = []
    for i in range(0,100):
        n = random.randint(10,18)
        random_list.append(n)
    return random_list

def average_filter(list, new_n):
    l = len(list)
    if l:
        list.append(avg(list)*l/(l+1)+new_n/(l+1))
    else:
        list.append(new_n)
                
def avg(list):
    return sum(list)/len(list)

random_list = generate_random_list()
avg_filter_list = []
for i in range(100):
    average_filter(avg_filter_list, random_list[i])


#plotting the random list
import matplotlib.pyplot as plt




plt.plot(random_list, 'bo', color='blue', linestyle='dashed', label='Random List', linewidth=2)
plt.suptitle('Random List', fontsize=14, fontweight='bold')
plt.plot(avg_filter_list, 'ro',color='red', linestyle='dashed', label='Average Filter List', linewidth=2)
plt.show()