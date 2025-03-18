import boto3

def terminate_ec2_instance(instance_id):
    # Create an EC2 resource
    ec2 = boto3.resource('ec2')

    # Retrieve the instance to be terminated
    instance = ec2.Instance(instance_id)

    # Terminate the EC2 instance
    response = instance.terminate()

    # Wait for the instance to terminate
    instance.wait_until_terminated()

    print("Terminated instance:", instance_id)

if __name__ == "__main__":
    # Replace with your instance ID
    my_instance_id = 'i-033e2043a62347876'
    terminate_ec2_instance(my_instance_id)