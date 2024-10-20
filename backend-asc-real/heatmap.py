thres = 0.8 - 0.01
ind_layer_to_del = set()  # Store indices of layers to delete.

# Loop over each layer 'i'
for i in range(len(avg_hidden_state_outputs)):
    # Skip if the layer is already marked for deletion.
    if i in ind_layer_to_del:
        continue

    # Compare layer 'i' with all subsequent layers 'k'
    for k in range(i + 1, len(avg_hidden_state_outputs)):
        if k in ind_layer_to_del:  # Skip deleted layers.
            continue

        # If similarity is above threshold, mark 'k' for deletion.
        if avg_hidden_state_outputs[i][k] >= thres:
            ind_layer_to_del.add(k)  # Mark layer 'k' for deletion.
            print(avg_hidden_state_outputs[i][k])

# Print the sorted list of deleted layers.
print("Layers to delete:", sorted(ind_layer_to_del))

ind_layer_to_del = [i-1 for i in ind_layer_to_del]

print("Index of Layers to delete:", sorted(ind_layer_to_del))