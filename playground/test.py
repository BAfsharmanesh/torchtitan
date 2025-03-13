import time

print("This is a test")

# sleep for 10 seconds
def sleep_me(i):
    sl_time = 10+i*2
    print(f"Process {i} is sleeping for {sl_time} seconds")
    time.sleep(sl_time)

# spawn 5 python processes in parallel that call sleep_me
import multiprocessing
processes = []
for i in range(5):
    p = multiprocessing.Process(target=sleep_me, args=(i,))
    processes.append(p)
    p.start()
    print(f"Started process {i}, pid={p.pid}")
    
for p in processes:    
    p.join()
    print(f"Joined process {p.pid}")

print("All processes have completed")

