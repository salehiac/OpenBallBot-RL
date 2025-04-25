import matplotlib.pyplot as plt
import json
import sys

with open(sys.argv[1],"r") as fl:
    Gs=json.load(fl)


limit=int(sys.argv[2]) if len(sys.argv)==3 else -1


plt.plot(Gs[:limit])
plt.grid("on")
plt.show()
