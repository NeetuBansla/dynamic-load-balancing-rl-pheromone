import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, optim
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import random

from preprocessing.preprocessing import preprocess_data

import os

RUN_SEED = int(os.environ.get("RUN_SEED", 42))

random.seed(RUN_SEED)
np.random.seed(RUN_SEED)
torch.manual_seed(RUN_SEED)

# ---------------------------------------------
# DATA PREPROCESSING
# ---------------------------------------------
df_clean = preprocess_data("borg_traces_data.csv")

# -------------------------------------------------------
# TASK PROFILER MODULE
# Extract task type, size, urgency, and complexity
# -------------------------------------------------------
def task_profiler(row):
    """
    Profile a task and return its characteristics.
    """
    task_type = "light" if row["req_cpu"] < 2 else "heavy"
    task_size = row["req_cpu"] * row["req_memory"]
    task_priority = row["priority"]
    
    return {
        "task_type": task_type,
        "task_size": task_size,
        "priority": task_priority,
        "req_cpu": row["req_cpu"],
        "req_memory": row["req_memory"],
        "avg_cpu_usage": row["avg_cpu_usage"],
        "max_cpu_usage": row["max_cpu_usage"]
    }

# ---------------------------------------------
# Resource Monitor, PSFOA, RL Scheduler
# ---------------------------------------------

# Define nodes (Edge + Cloud)
nodes = {
    "edge_1": {"type": "edge", "cpu_total": 8, "mem_total": 16000, "cpu_used": 0, "mem_used": 0},
    "edge_2": {"type": "edge", "cpu_total": 8, "mem_total": 16000, "cpu_used": 0, "mem_used": 0},
    "cloud_1": {"type": "cloud", "cpu_total": 64, "mem_total": 256000, "cpu_used": 0, "mem_used": 0},
    "cloud_2": {"type": "cloud", "cpu_total": 64, "mem_total": 256000, "cpu_used": 0, "mem_used": 0}
}

def get_node_status():
    """Return current node status."""
    status = {}
    for node, info in nodes.items():
        status[node] = {
            "cpu_available": info["cpu_total"] - info["cpu_used"],
            "mem_available": info["mem_total"] - info["mem_used"],
            "cpu_total": info["cpu_total"],       # <-- added
            "mem_total": info["mem_total"],       # <-- added
            "type": info["type"]
        }
    return status

def psfoa_suggestion(task, w_cpu=0.6, w_mem=0.4, exploration=0.05):
    """
    Return candidate nodes based on available resources with weighted scoring.
    Includes a small random exploration factor.
    """
    status = get_node_status()
    candidates = []
    
    for node, info in status.items():
        # Check if node can handle task
        if info["cpu_available"] >= task["req_cpu"] and info["mem_available"] >= task["req_memory"]:
            
            # Weighted score based on available CPU and Memory
            score = (
                w_cpu * (info["cpu_available"] / info["cpu_total"]) +
                w_mem * (info["mem_available"] / info["mem_total"])
            )

            
            # Small bonus if node type matches Edge/Cloud label
            if task.get("label_edge_cloud", 0) == 1 and info["type"] == "edge":
                score *= 1.05  # 5% bonus for preferred node
            elif task.get("label_edge_cloud", 0) == 0 and info["type"] == "cloud":
                score *= 1.05
            
            # Add small random exploration factor
            score *= 1 + random.uniform(0, exploration)
            
            candidates.append((node, score))
    
    # Sort candidates by descending score
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    return [node for node, score in candidates]

class RLScheduler:
    def __init__(self):
        self.q_table = {}  # state-action values
    
    def get_state(self, task):
        return (task["req_cpu"], task["req_memory"], task["priority"])
    
    def get_q(self, state, node):
        return self.q_table.get((state, node), 0)
    
    def choose_node(self, task, candidates):
        state = self.get_state(task)
        
        # ε-greedy policy
        if random.random() < 0.01:
            return random.choice(candidates)
        
        return max(candidates, key=lambda n: self.get_q(state, n))
    
    def update_reward(self, task, node, reward):
        state = self.get_state(task)
        key = (state, node)
        old_q = self.q_table.get(key, 0)
        
        # simple Q-learning
        self.q_table[key] = old_q + 0.1 * (reward - old_q)

scheduler = RLScheduler()

def orchestrator_decision(task_features):
    task_features = scaler.transform([task_features])[0]
    task_tensor = torch.tensor(task_features, dtype=torch.float32).unsqueeze(0)
    pred = model(task_tensor)
    return 1 if pred.item() > 0.5 else 0

def schedule_task(task):

    # If label was not provided, compute it
    if "label_edge_cloud" not in task:
        task_label = orchestrator_decision([
            task["priority"], 
            task["scheduling_class"], 
            task["req_cpu"], 
            task["req_memory"], 
            task["avg_cpu_usage"], 
            task["max_cpu_usage"]
        ])
        task["label_edge_cloud"] = task_label

    # Get ranked candidate nodes (soft bias included)
    candidates = psfoa_suggestion(task)

    # DO NOT FILTER BY TYPE ANYMORE
    # candidates remain as PSFOA sorted list

    if not candidates:
        return "REJECTED"

    # RL chooses best from candidate list
    selected_node = scheduler.choose_node(task, candidates)

    # Update node resource usage
    nodes[selected_node]["cpu_used"] += task["req_cpu"]
    nodes[selected_node]["mem_used"] += task["req_memory"]

    return selected_node

# Features (excluding categorical for simplicity, or one-hot encode later)
X = df_clean[["priority", "scheduling_class", "req_cpu", "req_memory", 
              "avg_cpu_usage", "max_cpu_usage"]].values
y = df_clean["label_edge_cloud"].values


# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=RUN_SEED,
    stratify=y
)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

class CIFGLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Combined input-forget gate
        self.xh = nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size)
    
    def forward(self, x, hc):
        h_prev, c_prev = hc
        combined = torch.cat([x, h_prev], dim=1)
        
        # Coupled input-forget gate: f = 1 - i
        i = torch.sigmoid(self.xh(combined))
        f = 1 - i
        
        g = torch.tanh(self.candidate(combined))
        c_next = f * c_prev + i * g
        h_next = torch.tanh(c_next)
        return h_next, c_next

class CIFGLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = CIFGLSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Initialize hidden and cell states
        h = torch.zeros(x.size(0), self.hidden_size)
        c = torch.zeros(x.size(0), self.hidden_size)
        h, c = self.cell(x, (h, c))
        out = self.fc(h)
        out = self.sigmoid(out)
        return out

model = CIFGLSTM(input_size=X_train.shape[1], hidden_size=32)
# Compute class weight: more weight for minority class
pos_weight = torch.tensor([len(y_train[y_train==0]) / len(y_train[y_train==1])], dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.Adam(model.parameters(), lr=0.00003)

import time

start_time = time.time()

epochs = 35
batch_size = 128

for epoch in range(epochs):
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0
    
    for i in range(0, X_train.size(0), batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)
    
    epoch_loss /= X_train.size(0)
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")


end_time = time.time()

training_time = end_time - start_time
print(f"\nTraining Time: {training_time:.4f} seconds")

failover_times = []

def fault_tolerant_schedule(task_features, max_retries=3):
    global failover_times, total_migrations, successful_migrations
    retries = 0
    assigned = "REJECTED"

    while retries < max_retries:
        task_label = orchestrator_decision(task_features)
        task_dict = {
            "req_cpu": task_features[2],
            "req_memory": task_features[3],
            "priority": task_features[0],
            "scheduling_class": task_features[1],
            "avg_cpu_usage": task_features[4],
            "max_cpu_usage": task_features[5],
            "label_edge_cloud": task_label
        }

        assigned = schedule_task(task_dict)

        # ✅ Migration counting
        if retries > 0:
            total_migrations += 1
            if assigned != "FAILED":
                successful_migrations += 1

        if assigned != "REJECTED":
            failover_times.append(retries + 1)
            return assigned

        retries += 1

        if retries == max_retries:
            all_nodes = psfoa_suggestion({"req_cpu": task_features[2], "req_memory": task_features[3]})
            if all_nodes:
                failover_times.append(retries + 1)
                return all_nodes[0]

    failover_times.append(max_retries)
    return "FAILED"

feedback_log = []  # initialize empty list

assigned_nodes = []

# ----------------------------------------------------------
# METRIC VARIABLES
# ----------------------------------------------------------

baseline_latencies = []
psfoa_latencies = []

completed_tasks = 0
total_tasks = 0

total_migrations = 0
successful_migrations = 0

correct_offload_predictions = 0

# Node load variance tracking
load_history = []

overhead_times = []

# Loop through each task in the cleaned dataframe
for i, row in tqdm(df_clean.iterrows(), total=df_clean.shape[0], desc="Scheduling Tasks"):
    
    
    # -------------------------------
    # RESET NODE USAGE PERIODICALLY
    # -------------------------------
    if i % 5000 == 0 and i > 0:
        for n in nodes:
            nodes[n]["cpu_used"] *= 0.5   # reduce load to simulate task completion
            nodes[n]["mem_used"] *= 0.5

    # Extract task features for orchestrator
    task_features = [
        row["priority"], 
        row["scheduling_class"], 
        row["req_cpu"], 
        row["req_memory"], 
        row["avg_cpu_usage"], 
        row["max_cpu_usage"]
    ]
    
    # Schedule task with fault tolerance
    profile = task_profiler(row)

    node = fault_tolerant_schedule(task_features)
    assigned_nodes.append(node)
    
    total_tasks += 1
    predicted_label = orchestrator_decision(task_features)
    
    # ---------------------------
    # 1. PSFOA LATENCY
    # ---------------------------
    if node != "FAILED" and node != "REJECTED":
        latency = row["req_cpu"] * 0.2 + row["req_memory"] * 0.001   # example formula
        psfoa_latencies.append(latency)
        completed_tasks += 1
        
        baseline_latency = row["req_cpu"] * 0.25 + row["req_memory"] * 0.002
        baseline_latencies.append(baseline_latency)

    
    # ---------------------------
    # 2. OFFLOADING ACCURACY
    # ---------------------------
    if node in nodes:
        actual_label = row["label_edge_cloud"]   # correct ground truth
        if predicted_label == actual_label:
            correct_offload_predictions += 1

    
    # ---------------------------
    # 3. LOAD VARIANCE TRACKING
    # ---------------------------
    current_loads = [
        nodes[n]["cpu_used"] / nodes[n]["cpu_total"]
        for n in nodes
    ]
    load_history.append(np.var(current_loads))

    # Log feedback
    feedback_log.append({
        "task_id": i,
        "assigned_node": node,
        "status": "success" if node != "FAILED" else "failed",
        "req_cpu": row["req_cpu"],
        "req_memory": row["req_memory"],
        "priority": row["priority"]

    })
    
    start = time.time()
    
    node = fault_tolerant_schedule(task_features)
    
    end = time.time()
    
    overhead_times.append(end - start)

# Save results back to dataframe
df_clean["assigned_node"] = assigned_nodes

# Optional: convert feedback log to dataframe
feedback_df = pd.DataFrame(feedback_log)

print("\nFeedback log preview:")
print(feedback_df.head())

# -------------------------------------------------------
# FEEDBACK LEARNING LOOP (Self-Optimizing Scheduler)
# -------------------------------------------------------
def apply_feedback(feedback_df):
    for entry in feedback_df.to_dict("records"):
        task = {
            "req_cpu": entry["req_cpu"],
            "req_memory": entry["req_memory"],
            "priority": entry.get("priority", 1)
        }
        state = (task["req_cpu"], task["req_memory"], task["priority"])
        node = entry["assigned_node"]
        
        if node == "FAILED":
            reward = -5
        else:
            if node == "FAILED":
                reward = -3
            else:
                reward = 1 if nodes[node]["type"] == "edge" else 0.7

        
        scheduler.update_reward(task, node, reward)

apply_feedback(feedback_df)
print("Feedback learning applied.")


# Make predictions on test set
with torch.no_grad():
    THRESHOLD = 0.35  
    y_pred_probs = model(X_test)           # probabilities between 0 and 1
    #y_pred = (y_pred_probs > 0.5).int()    # threshold 0.5 -> 0 or 1
    y_pred = (y_pred_probs >= THRESHOLD).to(torch.int)


y_true = y_test.int()

y_true_np = y_true.numpy()
y_pred_np = y_pred.numpy()

report = classification_report(y_true_np, y_pred_np, target_names=["Cloud", "Edge"])
print(report)

# Compute metrics
accuracy = accuracy_score(y_true_np, y_pred_np)*100
precision = precision_score(y_true_np, y_pred_np)*100
recall = recall_score(y_true_np, y_pred_np)*100
f1 = f1_score(y_true_np, y_pred_np)*100

# Print metrics
print("\nModel Performance Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

avg_overhead = np.mean(overhead_times)
print(f"Average Scheduling Overhead: {avg_overhead:.6f} sec/task")

task_sizes = [1000, 5000, 10000, 20000]

scalability_results = []

for size in task_sizes:
    avg_latency = np.mean(psfoa_latencies[:size])
    success_rate = completed_tasks / total_tasks
    
    scalability_results.append((size, avg_latency, success_rate))

print("\n========== Scalability Results ==========\n")

for size, latency, success in scalability_results:
    print(f"Tasks: {size}")
    print(f"  Avg Latency   : {latency:.6f}")
    print(f"  Success Rate  : {success*100:.2f}%")
    print("--------------------------------------")

sizes = [r[0] for r in scalability_results]
latencies = [r[1] for r in scalability_results]

plt.figure()

plt.plot(
    sizes, 
    latencies, 
    marker='o',
    color='#1f77b4',        # 🔵 line color
    markerfacecolor='#ff7f0e',  # 🟠 marker fill
    markeredgecolor='black',
    linewidth=2
)

plt.xlabel("Number of Tasks")
plt.ylabel("Average Latency")
plt.title("Scalability: Tasks vs Latency")

# Add value labels
for x, y in zip(sizes, latencies):
    plt.text(x, y, f"{y:.4f}", color='black')

plt.grid(True, linestyle='--', alpha=0.6)

plt.show()


# Variables to export
__all__ = ["df_clean", "nodes", "accuracy", "precision", "recall", "f1", "X_test", "y_true_np", "y_pred_np","model" ,
           "feedback_log", "failover_times", "baseline_latencies", "psfoa_latencies",
           "completed_tasks", "total_migrations", "successful_migrations", "correct_offload_predictions",
           "training_time", "avg_overhead"]

