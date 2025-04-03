# AI Acceleration Prediction with Federated Learning

## Instructions for Running the Code
Download all provided files. Ensure that the .csv dataset files are in the same level as the server and client files. Run the server before the client. It will prompt you for method choice (FedAvg, FedDyn, IDA, Robust Aggregaton, or HyFDCA), number of rounds, noisy/clean dataset, and seed. After the server starts, you can run the client. The client file runs the client code a set number of times, incrementing the seed each time. When it finishes a .csv file will be created that has the mean squared error across each device per round.

### 1. Configure `Client.ipynb`
Before running the code, update the `num_iterations` variable in the `Client.ipynb` file as needed.  
> **Recommended:** Set to `10`, depending on your device's specifications.

---

### 2. Run `Server.ipynb`
When executing `Server.ipynb`, you will be prompted to select a method.  

> ⚠️ **Note:** The default is the previous run's input. For the **first run**, ensure you manually enter a value between `0-4`.

#### Method Options:
- `0` — **FedAvg**: Federated Averaging  
- `1` — **FedDyn**: Federated Learning with Dynamic Regularization  
- `2` — **IDA**: Inverse Distance Aggregation  
- `3` — **Robust**: Robust Federated Aggregation (Smoothed Weiszfeld Algorithm)  
- `4` — **HyFDCA**: Hybrid Federated Dual Coordinate Ascent  

---

### 3. Enter Number of Training Rounds
You will then be prompted to input the number of training rounds.  

> **Recommended:** `100+` training rounds for accurate results.

---

### 4. Choose Noisy Dataset Option
You will be asked whether you'd like to use a noisy dataset.  
> **Default:** `No`

---

### 5. Enter Seed Value
You will be prompted to input a seed value.  
> The seed auto-increments by 1 for every iteration and is used for ANOVA.

---

### 6. Run `Client.ipynb`
Lastly, run `Client.ipynb`. You will be asked again to enter the training method.  
