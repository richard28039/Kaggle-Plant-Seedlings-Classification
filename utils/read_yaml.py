import yaml


def r_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        hyper_file = yaml.safe_load(file)
    return hyper_file
