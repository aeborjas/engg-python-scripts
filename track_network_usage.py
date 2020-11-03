import time
import psutil

def main():
    old_value = 0    
    try:
        while True:
            new_value = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
            
            if old_value:
                send_stat((new_value - old_value,'SYSTEM'))

            old_value = new_value
            time.sleep(1)
    except KeyboardInterrupt:
        quit()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass

def convert_to_mbit(value):
    return value/1024./1024.*8

def send_stat(*args):
     
    text = '\t'.join([x for x in map(lambda x: f"{x[1]}= {convert_to_mbit(x[0]):0.3f} MB/s",args)])
    print(text, end='\r')
main()
