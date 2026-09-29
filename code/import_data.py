from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
occupancy_detection = fetch_ucirepo(id=357) 
  
# data (as pandas dataframes) 
X = occupancy_detection.data.features 
y = occupancy_detection.data.targets 
  
# metadata 
print(occupancy_detection.metadata) 

# variable information 
print(occupancy_detection.variables) 


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(X['Temperature'], X['Humidity'], alpha=0.5)
plt.colorbar(label='Occupancy')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.title('Occupancy Detection: Temperature vs Humidity')
plt.show()