import matplotlib.pyplot as plt
import json
import sys

with open(sys.argv[1],"r") as fl:
    Gs=json.load(fl)


limit=int(sys.argv[2]) if len(sys.argv)==3 else len(Gs["avg_returns"])

if isinstance(Gs,dict):
    #plt.plot(Gs["total_steps"][:limit],Gs["avg_returns"][:limit])
    plt.plot(Gs["avg_returns"][:limit])
else:
    plt.plot(Gs[:limit])


#plt.plot(Gs["avg_returns"][:limit])
plt.grid("on")
plt.show()
