--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----
ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'cunn')
    if ok then
        print("warning: fbcunn not found. Falling back to cunn") 
        LookupTable = nn.LookupTable
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    LookupTable = nn.LookupTable
end
stringx = require('pl.stringx')
require('io')
require('nn')
--LookupTable = nn.LookupTable
require('nngraph')
require('base')
ptb = require('data')

-- Train 1 day and gives 82 perplexity.
--[[
local params = {batch_size=20,
                seq_length=35,
                layers=2,
                decay=1.15,
                rnn_size=1500,
                dropout=0.65,
                init_weight=0.04,
                lr=1,
                vocab_size=10000,
                max_epoch=14,
                max_max_epoch=55,
                max_grad_norm=10}
               ]]--

-- Trains 1h and gives test 115 perplexity.
local params = {batch_size=20,
                seq_length=20,
                layers=2,
                decay=2,
                rnn_size=200,
                dropout=0,
                init_weight=0.1,
                lr=1,
                vocab_size=10000,
                max_epoch=4,
                max_max_epoch=13,
                max_grad_norm=5}

function transfer_data(x)
  --return x
  return x:cuda()
end

--local state_train, state_valid, state_test
model = {}
--local paramx, paramdx

function lstm(i, prev_c, prev_h)
  local function new_input_sum()
    local i2h            = nn.Linear(params.rnn_size, params.rnn_size)
    local h2h            = nn.Linear(params.rnn_size, params.rnn_size)
    return nn.CAddTable()({i2h(i), h2h(prev_h)})
  end
  local in_gate          = nn.Sigmoid()(new_input_sum())
  local forget_gate      = nn.Sigmoid()(new_input_sum())
  local in_gate2         = nn.Tanh()(new_input_sum())
  local next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_gate2})
  })
  local out_gate         = nn.Sigmoid()(new_input_sum())
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end

function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local i                = {[0] = LookupTable(params.vocab_size,
                                                    params.rnn_size)(x)}
  local next_s           = {}
  local split         = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
  local dropped          = nn.Dropout(params.dropout)(i[params.layers])
  local pred             = nn.LogSoftMax()(h2y(dropped))
  --local pred_y           = nn.Max(2)(pred)
  local err              = nn.ClassNLLCriterion()({pred, y})
  local module           = nn.gModule({x, y, prev_s},
                                      {err, pred, nn.Identity()(next_s)})

  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end

function setup()
  print("Creating a RNN LSTM network.")
  local core_network = create_network()
  paramx, paramdx = core_network:getParameters()
  model.s = {}
  model.ds = {}
  model.start_s = {}
  for j = 0, params.seq_length do
    model.s[j] = {}
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length))
  model.pred = transfer_data(torch.zeros(params.seq_length, params.batch_size, params.vocab_size))
  --model.pred_y = transfer_data(torch.zeros(params.seq_length))
end

function reset_state(state)
  state.pos = 1
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

function fp(state)
  g_replace_table(model.s[0], model.start_s)
  if state.pos + params.seq_length > state.data:size(1) then
    reset_state(state)
  end
  for i = 1, params.seq_length do
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    model.err[i], model.pred[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s}))
    state.pos = state.pos + 1
  end
  g_replace_table(model.start_s, model.s[params.seq_length])
  return model.err:mean()
end

function bp(state)
  paramdx:zero()
  reset_ds()
  for i = params.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    --print(y)
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local dpred = transfer_data(torch.zeros(params.batch_size, params.vocab_size))
    --local dpred_y = transfer_data(torch.zeros())
    local tmp = model.rnns[i]:backward({x, y, s},
                                       {derr, dpred, model.ds})[3] 
    g_replace_table(model.ds, tmp)
    cutorch.synchronize()
  end
  state.pos = state.pos + params.seq_length
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
  paramx:add(paramdx:mul(-params.lr))
end

function run_valid()
  reset_state(state_valid)
  g_disable_dropout(model.rnns)
  local len = (state_valid.data:size(1) - 1) / (params.seq_length)
  local perp = 0
  for i = 1, len do
    perp = perp + fp(state_valid)
  end
  print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
  g_enable_dropout(model.rnns)
end

function run_test()
  reset_state(state_test)
  g_disable_dropout(model.rnns)
  local perp = 0
  local len = state_test.data:size(1)
  g_replace_table(model.s[0], model.start_s)
  for i = 1, (len - 1) do
    local x = state_test.data[i]
    print(x)
    local y = state_test.data[i + 1]
    local s = model.s[i - 1]
    perp_tmp, _, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    perp = perp + perp_tmp[1]
    g_replace_table(model.s[0], model.s[1])
  end
  print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
  g_enable_dropout(model.rnns)
end

function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  if tonumber(line[1]) == nil then error({code="init"}) end
  for i = 2,#line do
    if ptb.vocab_map[line[i]] == nil then error({code="vocab", word = line[i]}) end
  end
  return line
end

--Taken from https://github.com/rlowrance/re/blob/master/argmax.lua
local function argmax_1D(v)
   local length = v:size(1)
   assert(length > 0)

   -- examine on average half the entries
   local maxValue = torch.max(v)
   for i = 1, v:size(1) do
      if v[i] == maxValue then
         return i
      end
   end
end

function qs_input()
  while true do
    print("Query: len word1 word2 etc")
    local ok, line = pcall(readline)
    if not ok then
      if line.code == "EOF" then
        break -- end loop
      elseif line.code == "vocab" then
        print("Word not in vocabulary, only 'foo' is in vocabulary: ", line.word)
      elseif line.code == "init" then
        print("Start with a number")
      else
        print(line)
        print("Failed, try again")
      end
    else
      return line
      --[[
      print("Thanks, I will print foo " .. line[1] .. " more times")
      for i = 1, line[1] do io.write('foo ') end
      io.write('\n')
      ]]--
    end
  end
end

function query_sentences()
  line = qs_input()
  predict_num = table.remove(line, 1)
  print(predict_num)
  local len = table.getn(line)
  words = torch.Tensor(len)
  for j = 1, len do
    words[j] = ptb.vocab_map[line[j]]
  end
  state_query = {data=words}
  --print(state_query)
  --mask = torch.ByteTensor(1, line_tensor:size())
  --data = line_tensor
  reset_state(state_query)
  g_disable_dropout(model.rnns)
  print(state_query)
  local pred = torch.ones(params.vocab_size)
  g_replace_table(model.s[0], model.start_s)
  for i = 1, (len - 1) do
    local x = state_query.data[i]
    print(x)
    local y = state_query.data[i + 1]
    local s = model.s[i - 1]
    _, pred, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    g_replace_table(model.s[0], model.s[1])
  end
  local prev = state_query.data[len]
  local sentence = {}
  for i = len, (len + predict_num) do
    local s = model.s[i - 1]
    sentence[i - len] = prev
    _, pred, model.s[1] = unpack(model.rnns[1]:forward({prev, pred, model.s[0]}))
    prev = argmax(pred)
  end
  print("Thanks, I will print foo " .. line[1] .. " more times")
  for i = 1, sentence:size() do io.write(ptb.inv_vocab_map[sentence[i]], ' ') end
  io.write('\n')
  g_enable_dropout(model.rnns)
end

--[[
--function main()
g_init_gpu(arg)
state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}
print("Network parameters:")
print(params)
local states = {state_train, state_valid, state_test}
for _, state in pairs(states) do
 reset_state(state)
end
setup()
step = 0
epoch = 0
total_cases = 0
beginning_time = torch.tic()
start_time = torch.tic()
print("Starting training.")
words_per_step = params.seq_length * params.batch_size
epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
--perps
while epoch < params.max_max_epoch do
 perp = fp(state_train)
 if perps == nil then
   perps = torch.zeros(epoch_size):add(perp)
 end
 perps[step % epoch_size + 1] = perp
 step = step + 1
 bp(state_train)
 total_cases = total_cases + params.seq_length * params.batch_size
 epoch = step / epoch_size
 if step % torch.round(epoch_size / 10) == 10 then
   wps = torch.floor(total_cases / torch.toc(start_time))
   since_beginning = g_d(torch.toc(beginning_time) / 60)
   print('epoch = ' .. g_f3(epoch) ..
         ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
         ', wps = ' .. wps ..
         ', dw:norm() = ' .. g_f3(model.norm_dw) ..
         ', lr = ' ..  g_f3(params.lr) ..
         ', since beginning = ' .. since_beginning .. ' mins.')
 end
 if step % epoch_size == 0 then
   run_valid()
   if epoch > params.max_epoch then
       params.lr = params.lr / params.decay
   end
 end
 if step % 33 == 0 then
   cutorch.synchronize()
   collectgarbage()
 end
 --torch.save("lstm_model", model)
end
run_test()
torch.save("lstm_model", model)
torch.save("lstm_vocab_map", ptb.vocab_map)
torch.save("lstm_inv_vocab_map", ptb.inv_vocab_map)
print("Training is over.")
]]--
ptb.inv_vocab_map = torch.load("./lstm_inv_vocab_map")
ptb.vocab_map = torch.load("./lstm_vocab_map")
model = torch.load("./lstm_model")
state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}
run_test()
query_sentences()
--end