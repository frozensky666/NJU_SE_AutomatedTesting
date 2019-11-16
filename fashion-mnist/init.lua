local torch = require 'torch'
require 'paths'

local fashion_mnist = {}

local function readlush(filename)
   local f = torch.DiskFile(filename)
   f:bigEndianEncoding()
   f:binary()
   local ndim = f:readInt() - 0x800
   assert(ndim > 0)
   local dims = torch.LongTensor(ndim)
   for i=1,ndim do
      dims[i] = f:readInt()
      assert(dims[i] > 0)
   end
   local nelem = dims:prod(1):squeeze()
   local data = torch.ByteTensor(dims:storage())
   f:readByte(data:storage())
   f:close()
   return data
end

local function createdataset(dataname, labelname)
   local data = readlush(dataname)
   local label = readlush(labelname)
   assert(data:size(1) == label:size(1))
   local dataset = {data=data, label=label, size=data:size(1)}
   setmetatable(dataset, {__index=function(self, idx)
                                     assert(idx > 0 and idx <= self.size)
                                     return {x=data[idx], y=label[idx]}
                                  end})
   return dataset
end


function fashion_mnist.traindataset()
   local path = paths.dirname( paths.thisfile() )
   return createdataset(paths.concat(path, 'data/train-images-idx3-ubyte'),
                        paths.concat(path, 'data/train-labels-idx1-ubyte'))
end

function fashion_mnist.testdataset()
   local path = paths.dirname( paths.thisfile() )
   return createdataset(paths.concat(path, 'data/t10k-images-idx3-ubyte'),
                        paths.concat(path, 'data/t10k-labels-idx1-ubyte'))

end

return fashion_mnist
