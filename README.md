# 🚀 Project Execution Guide  
**Adaptive Load Balancing in Wireless Mesh Networks using Machine Learning**

Follow the steps below to execute the project from simulation to visualization:

---

## 🔧 Step 1: Run Initial NS-3 Simulation
- Execute the `wireless-mesh-custom.cc` file using your Ubuntu-based NS-3 environment.
- This will generate a file named `simulation_data.csv` containing key performance metrics.

---

## 📊 Step 2: Perform ML Model Selection (Google Colab)
- Upload the generated `simulation_data.csv` file to your Google Colab environment.
- Open and run the `ML_model_selection.ipynb` notebook.
- This notebook compares multiple machine learning models to identify the most effective one for congestion prediction.

---

## 🧠 Step 3: Finalize and Reconfigure ML Model in NS-3
- Once the best ML model is selected, it is re-integrated into the `adaptive-lb-simulation.cc` file.
- Run this updated simulation script in NS-3 to apply adaptive load balancing.

---

## 📁 Step 4: Generate Node Statistics
- In a Python environment, execute the `node_stats_generator.py` script.
- This will generate `node_stats.csv`, which contains detailed statistics for each node in the network.

---

## 📈 Step 5: Launch Visualization Dashboard
- Run the `mesh_network_dashboard.py` script.
- Upload both `simulation_data.csv` and `node_stats.csv` when prompted.
- The dashboard will provide interactive visualizations for:
  - Network overview
  - Node-wise statistics
  - Congestion levels
  - Adaptive balancing performance

---

## ✅ Output Summary
By following these steps, you will:
- Simulate a wireless mesh network in NS-3
- Apply machine learning to dynamically adapt to congestion
- Visualize the network behavior with real-time insights
