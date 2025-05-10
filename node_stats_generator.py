import csv

# Copy and paste the console output between the triple quotes
console_output = """Node 0: BW=0Mbps, Loss=0, PDR=1, Congestion=16.085
Node 1: BW=0.354413Mbps, Loss=2389, PDR=0.677747, Congestion=16.055
Node 2: BW=0Mbps, Loss=0, PDR=1, Congestion=16.085
Node 3: BW=0.344849Mbps, Loss=2242, PDR=0.709998, Congestion=15.975
Node 4: BW=0Mbps, Loss=0, PDR=1, Congestion=16.085
Node 5: BW=0Mbps, Loss=0, PDR=1, Congestion=16.085
Node 6: BW=0.376156Mbps, Loss=1554, PDR=0.757504, Congestion=15.975
Node 7: BW=0.382062Mbps, Loss=1987, PDR=0.703051, Congestion=15.975
Node 8: BW=0Mbps, Loss=0, PDR=1, Congestion=16.085
Node 9: BW=0Mbps, Loss=0, PDR=1, Congestion=16.085
Node 10: BW=0Mbps, Loss=0, PDR=1, Congestion=16.085
Node 11: BW=0.343626Mbps, Loss=1882, PDR=0.733193, Congestion=15.975 (FAILED)
Node 12: BW=0Mbps, Loss=0, PDR=1, Congestion=16.085
Node 13: BW=0.343596Mbps, Loss=2290, PDR=0.702059, Congestion=15.975
Node 14: BW=0.34409Mbps, Loss=1677, PDR=0.773505, Congestion=15.975
Node 15: BW=0Mbps, Loss=0, PDR=1, Congestion=16.085
Node 16: BW=0Mbps, Loss=0, PDR=1, Congestion=16.085
Node 17: BW=0.343615Mbps, Loss=755, PDR=0.893823, Congestion=15.975
Node 18: BW=0Mbps, Loss=0, PDR=1, Congestion=16.085
Node 19: BW=0Mbps, Loss=0, PDR=1, Congestion=16.085
Node 20: BW=0Mbps, Loss=0, PDR=1, Congestion=16.085
Node 21: BW=0Mbps, Loss=0, PDR=1, Congestion=16.085
Node 22: BW=0Mbps, Loss=0, PDR=1, Congestion=16.085
Node 23: BW=0Mbps, Loss=0, PDR=1, Congestion=16.085
Node 24: BW=0Mbps, Loss=0, PDR=1, Congestion=16.085 (FAILED)"""

# Parse the console output
node_stats = []
for line in console_output.strip().split('\n'):
    # Extract node ID
    node_id = int(line.split(':')[0].replace('Node ', ''))
    
    # Extract bandwidth
    bw_str = line.split('BW=')[1].split('Mbps')[0]
    bandwidth = float(bw_str)
    
    # Extract packet loss
    loss_str = line.split('Loss=')[1].split(',')[0]
    packet_loss = int(loss_str)
    
    # Extract PDR
    pdr_str = line.split('PDR=')[1].split(',')[0]
    pdr = float(pdr_str)
    
    # Extract congestion
    congestion_str = line.split('Congestion=')[1].split()[0]
    congestion = float(congestion_str.replace('(FAILED)', '').strip())
    
    # Check if node has failed
    has_failed = 1 if '(FAILED)' in line else 0
    
    # Check if node is overloaded (congestion > 16.2)
    is_overloaded = 1 if congestion > 16.2 else 0
    
    node_stats.append([
        node_id, 
        bandwidth, 
        packet_loss, 
        pdr, 
        congestion, 
        is_overloaded, 
        has_failed
    ])

# Write to CSV
with open('node_stats.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['nodeId', 'bandwidthUtilization', 'packetLossCount', 
                    'packetDeliveryRatio', 'congestionRatio', 'isOverloaded', 'hasFailed'])
    writer.writerows(node_stats)

print("Successfully created node_stats.csv")

# Print the first few rows to verify
print("\nPreview of node_stats.csv:")
for i in range(min(5, len(node_stats))):
    print(f"Node {node_stats[i][0]}: BW={node_stats[i][1]}Mbps, Loss={node_stats[i][2]}, PDR={node_stats[i][3]}, " +
          f"Congestion={node_stats[i][4]}, Overloaded={node_stats[i][5]}, Failed={node_stats[i][6]}")