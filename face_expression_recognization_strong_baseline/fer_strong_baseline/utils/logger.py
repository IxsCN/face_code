import  os
import numpy as np
import yacs
import pdb

class TxtLogger(object):
    def __init__(self, output_name, print_on_screen=True):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}
        self.print_on_screen = print_on_screen

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, *msgs):
        for msg in msgs:
            # if isinstance(msg, str):
            #     self.log_file.write(msg + '\n')
            # elif isinstance(msg, list):
            #     self.write(self, *msg)
            # elif isinstance(msg, np.ndarray):
            #     self.log_file.write(msg.__str__())
            # elif isinstance(msg, float) or isinstance(msg, int) or isinstance(msg, bool):
            #     self.log_file.write(str(msg) + '\n')
            # elif isinstance(msg, yacs.config.CfgNode):
            #     self.log_file
            if '__str__'in dir(msg):
                if self.print_on_screen:
                    print(msg)
                self.log_file.write(msg.__str__() + '\n')
            else:
                print('not support: ', type(msg))
                raise NotImplementedError
        self.log_file.flush()

    def close(self):
        self.log_file.close()

if __name__ == '__main__':
    print(type(np.array([1,2,3])))