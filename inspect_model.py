import tensorflow as tf
import numpy as np
import h5py
import json

def decode_if_bytes(value):
    """Decode bytes to utf-8 if needed"""
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value

def inspect_h5_model(filepath):
    """Inspect the structure of an H5 model file"""
    with h5py.File(filepath, 'r') as f:
        print("\nModel structure:")
        def print_attrs(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"\nGroup: {name}")
                if 'layer_names' in obj.attrs:
                    layer_names = [decode_if_bytes(n) for n in obj.attrs['layer_names']]
                    print("Layer names:", layer_names)
                if 'backend' in obj.attrs:
                    print("Backend:", decode_if_bytes(obj.attrs['backend']))
                for key in obj.attrs.keys():
                    if key not in ['layer_names', 'backend', 'keras_version']:
                        print(f"Attribute: {key}")
                        try:
                            value = obj.attrs[key]
                            if isinstance(value, (bytes, np.ndarray)):
                                value = [decode_if_bytes(v) for v in value]
                            print(f"Value: {value}")
                        except:
                            print("Value: <unable to decode>")
        
        # Print all groups and datasets
        f.visititems(print_attrs)
        
        # Print model weights structure
        if 'model_weights' in f:
            print("\nDetailed model weights structure:")
            for layer_name in f['model_weights']:
                print(f"\nLayer: {layer_name}")
                layer_group = f['model_weights'][layer_name]
                if hasattr(layer_group, 'attrs'):
                    for attr_name, attr_value in layer_group.attrs.items():
                        try:
                            if isinstance(attr_value, (bytes, np.ndarray)):
                                value = [decode_if_bytes(v) for v in attr_value]
                                print(f"  {attr_name}: {value}")
                            else:
                                print(f"  {attr_name}: {attr_value}")
                        except:
                            print(f"  {attr_name}: <unable to decode>")
                
                # Print weight shapes
                if hasattr(layer_group, 'items'):
                    for weight_name, weight_data in layer_group.items():
                        if isinstance(weight_data, h5py.Dataset):
                            print(f"  Weight {weight_name}: shape={weight_data.shape}, dtype={weight_data.dtype}")

if __name__ == "__main__":
    # Inspect January model
    model_path = "lstm_vae_model/lstm_vae_model_jan.h5"
    print(f"\nInspecting model: {model_path}")
    inspect_h5_model(model_path)
