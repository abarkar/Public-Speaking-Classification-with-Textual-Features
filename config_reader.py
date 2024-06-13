import configparser
import os

def read_config(config_file='config.ini'):
    config = configparser.ConfigParser()
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} does not exist.")
    
    config.read(config_file)
    
    # Default values can be provided here
    default_settings = {
        'rootDirPath': '/home/alisa/Documents/GitHubProjects/Public-Speaking-Classification-with-Textual-Features',
        'dataset': 'MT',
        'dimension': ['persuasiveness'],
        'clip': 'full',
        'model': ['SVM'],
        'clasSeparator': 'mean',
        'aggregationMethod': 'rms',
        'task': 'classification',
        'modalities': ['text'],
        'threshold': 0.10,
        'featureSelection': True
    }

    # Helper function to convert comma-separated strings to lists
    def to_list(value):
        return value.split(',') if isinstance(value, str) else value

    # Helper function to convert string to boolean
    def to_bool(value):
        return value.lower() in ('true', '1', 'yes') if isinstance(value, str) else value

    settings = {}
    for key, default in default_settings.items():
        if key in config['Settings']:
            value = config['Settings'][key]
            if key in ['dimension', 'model', 'modalities']:
                settings[key] = to_list(value)
            elif key == 'threshold':
                settings[key] = float(value)
            elif key == 'featureSelection':
                settings[key] = to_bool(value)
            else:
                settings[key] = value
        else:
            settings[key] = default
    
    return settings

# Example usage
if __name__ == "__main__":
    try:
        config = read_config()
        print(config)
    except Exception as e:
        print(f"Error reading configuration: {e}")
