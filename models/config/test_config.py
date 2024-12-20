import yaml

test_data = {
    'cameras': [{
        'id': 1,
        'ip': "192.168.1.2"
    }, {
        'id': 2,
        'ip': "192.168.1.3"
    }]
}


def read_yaml(path):
    try:
        with open(path, 'r') as file:
            data = file.read()
            # result = yaml.load(data)
            result = yaml.load(data, Loader=yaml.FullLoader)

            return result
    except Exception as e:
        print(e)
        return None


def write_yaml(path):
    try:
        with open('path', 'w', encoding='utf-8') as f:
            yaml.dump(data=test_data, stream=f, allow_unicode=True)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    p = 'train.yaml'
    result = read_yaml(p)
    # j=json.load(result)
    print('result', result)
    # print('cameras', result['cameras'])
    # print('json',j)

