"""
Get dog breed's information from dynamodb (local).

Before use this, start DynamoDB Local like below.
java -Djava.library.path=./DynamoDBLocal_lib -jar DynamoDBLocal.jar -dbPath ./dir/path/to/db
"""

import boto3
import decimal
import json
from urllib.parse import unquote


class Dynamodb(object):
    def __init__(self):
        self.dynamodb = boto3.resource(
            'dynamodb',
            region_name='breeds',
            endpoint_url="http://localhost:8000",
            aws_access_key_id='dog',
            aws_secret_access_key='_'
        )

        self.breeds = []
        with open('../model/breeds_16.txt') as f:
            lines = f.readlines()
            for line in lines:
                self.breeds.append(line.replace('\n', ''))

    def create_table(self, table_name):
        """
        Create database table.

        Args:
            table_name (str): Table's name
        """

        table = self.dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {
                    'AttributeName': 'breed_id',
                    'KeyType': 'HASH'  # Partition key
                },
                {
                    'AttributeName': 'breed_name',
                    'KeyType': 'RANGE'  # Sort key
                }
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'breed_id',
                    'AttributeType': 'N'
                },
                {
                    'AttributeName': 'breed_name',
                    'AttributeType': 'S'
                },
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 10,
                'WriteCapacityUnits': 10
            }
        )
        print('Table status:', table.table_status)

    def add_data(self, table_name, json_path):
        """
        Add data to database.
        FIXME: Why multi byte charactor causes bugs?
            Now, encording requires in json file.
            Tool: https://tech-unlimited.com/urlencode.html

        Args:
            table_name (str): Table's name
            json_path (str): Json file path written data (breed_id, breed_name, description).
        """

        table = self.dynamodb.Table(table_name)

        with open(json_path) as json_file:
            breeds = json.load(json_file, parse_float=decimal.Decimal)
            for breed in breeds:
                breed_id = int(breed['breed_id'])
                breed_name = breed['breed_name']
                description = breed['description']

                print("Adding breed:", breed_id, breed_name)

                table.put_item(
                    Item={
                        'breed_id': breed_id,
                        'breed_name': breed_name,
                        'description': description,
                    }
                )

    def load_data(self, table_name, breed_id):
        """
        Load data from database.

        Args:
            table_name (str): Table's name
            breed_id (int): Index of targe keyword
        """

        table = self.dynamodb.Table(table_name)
        breed_name = self.breeds[breed_id]

        try:
            response = table.get_item(
                Key={
                    'breed_id': breed_id,
                    'breed_name': breed_name
                }
            )
        except ClientError as e:
            return e.response['Error']['Message']
        else:
            item = response['Item']
            return unquote(item['description']).replace('+', ' ')


if __name__ == "__main__":
    dynamodb = Dynamodb()

    # dynamodb.create_table('Breeds')
    # dynamodb.add_data('Breeds', 'data.json')

    print(dynamodb.load_data('Breeds', 4))
    print(dynamodb.load_data('Breeds', 10))
