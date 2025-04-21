import matplotlib.pyplot as plt
import json
import sys

with open(sys.argv[1],"r") as fl:
    Gs=json.load(fl)


plt.plot(Gs)
plt.show()
