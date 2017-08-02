import csv

class Record:
    def __init__(self, model_name, hidden_dim, word_dim, data_size, batch_size):
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.word_dim = word_dim
        self.data_size = data_size
        self.batch_size = batch_size
        self.fname = model_name + "-h" + str(hidden_dim) + "-vcb" + str(word_dim) + "-data" + str(data_size) + "-b" + str(batch_size) + ".csv"
        self.f = 0
        self.writer = 0
        
    def create(self):
        self.f = open("result/" + self.fname, "a")
        self.writer = csv.writer(self.f)
        self.writer.writerow(["epoch", "PPL"])

    def write(self, epoch, ppl):
        #g = open("result/" + self.fname, "w")
        #writer = csv.writer(g)
        self.writer.writerow([str(epoch), "{:,.2f}".format(ppl)])
        
    def close(self):
        self.f.close()
