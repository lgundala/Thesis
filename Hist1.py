import matplotlib.pyplot as plt
import random
x = random.sample(range(1000), 100)
xbins = [0, len(x)]
plt.bar(range(0,100), x)
plt.show()
print(x);
