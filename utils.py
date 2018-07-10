import shutil, os, pathlib, sys, time

class ElapsedTimer(object):
    def __init__(self,msg='Elapsed'):
        self.start_time = time.time()
        self.msg = msg
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print(self.msg + ": %s " % self.elapsed(time.time() - self.start_time),
              flush=True)
        return (self.msg + ": %s " % self.elapsed(time.time() - self.start_time))

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
NUM_CPU = cpu_count()

def parallel_process(array, function, n_jobs=NUM_CPU, use_kwargs=False, front_num=3):
    """
    A parallel version of the map function with a progress bar. 

    Args:
        array (array-like): An array to iterate over.
        function (function): A python function to apply to the elements of array
        n_jobs (int, default=16): The number of cores to use
        use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
            keyword arguments to function 
        front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
            Useful for catching bugs
    Returns:
        [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out

def help_option(msg):
    if '-h' in sys.argv or '--help' in sys.argv:
        print(msg)
        exit()

def safe_copytree(srcpath, dstpath, ignores):
    ''' 
    if dst_path doesn't exists,
    it creates new directory and copy all of contents under src
    except *ignores.

    src must exists.
    dst must not exists.

    # ex) copy directory structure without files.
    safe_copytree(src_dir_path, dst_dir_path, ['*.jpg', '*.jpeg', '*.png']) 
    '''
    try:
        shutil.copytree(srcpath, dstpath, 
                        ignore=shutil.ignore_patterns(*ignores))
    except Exception as e:
        print(e)

def safe_mkdir(directory_path):
    try:
        os.mkdir(directory_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def file_paths(root_dir_path):
    ''' generate file_paths of directory_path ''' 
    it = os.walk(root_dir_path)
    for root,dirs,files in it:
        for path in map(lambda name:os.path.join(root,name),files):
            yield path

def replace_part_of(srcpath, old_part, new_part):
    '''
    change old_part of srcpath with new_part
    old/new_part must be path or str(not include path delimiters) 
    '''
    p = pathlib.Path(srcpath)
    parts = [part.replace(old_part,new_part) for part in p.parts]
    return pathlib.Path(*parts)

def make_dstpath(srcpath, old_parent, new_ancestors):
    '''
    discard part of path until old_parent,
    and put new_ancestors into path

    ex) make_dstpath('a/b/cd/dx','cd','123/456') -> '123/456/dx'
    '''
    p = pathlib.Path(srcpath)
    idx = p.parts.index(old_parent)
    return str(pathlib.Path(new_ancestors) \
                      .joinpath(*p.parts[idx+1:]))

def slice1channel(gray_rgb_img, channel=0):
    ''' gray_rgb_img is r=g=b image. '''
    grayscale_img = gray_rgb_img[:,:,channel]
    return grayscale_img.reshape(gray_rgb_img.shape[:2] + (1,))

import unittest
class Test_replace_part_of_path(unittest.TestCase):
    def test_replace_part(self):
        p = pathlib.Path('root','bbb','ccc','leaf')
        self.assertEqual(replace_part_of(p,'root','xxxx'),
                         pathlib.Path('xxxx','bbb','ccc','leaf'))           
        self.assertEqual(replace_part_of(p,'leaf','xxxx'),
                         pathlib.Path('root','bbb','ccc','xxxx'))           
        self.assertEqual(replace_part_of(p,'bbb','123'),
                         pathlib.Path('root','123','ccc','leaf'))           

    def test_old_part_is_not_in_srcpath(self):
        # nothing happens.
        p = pathlib.Path('root','bbb','ccc','leaf')
        self.assertEqual(p,replace_part_of(p,'xx','asd'))

if __name__ == '__main__':
    print(make_dstpath('./asd/fde/gcd/sdf.jpg','gcd','./aaa/dx/'))
    print(make_dstpath('a/b/cd/dx','cd','123/456'))
    unittest.main()
