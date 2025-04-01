# AI_FL
# WILL BE UPDATED SOON TO REFLECT THE CHANGES TO CLIENT AND SERVER FILES

Run instructions:
Download all provided files. Ensure that the .csv dataset files are in the same level as the server and client files. Run the server before the client. It will prompt you for method choice (FedAvg, FedDyn, IDA, or Robust Aggregaton), number of rounds, noisy/clean dataset, and seed. After the server starts, you can run the client. The client file runs the client code a set number of times, incrementing the seed each time. When it finishes a .csv file will be created that has the mean squared error across each device per round.
