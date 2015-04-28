require 'torch'
require 'base'
ptb = require('data')
traindata = ptb.traindataset(20, true)
for k,v in pairs(ptb.vocab_map) do
  print(k .. " " .. v)
end
