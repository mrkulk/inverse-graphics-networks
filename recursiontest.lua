-- setvalue = function(tbl, key, value)
--   if tbl._parent then
--     setvalue(tbl._parent, 'modules', {[tbl._childIndex] = {[key] = value}})
--   elseif tbl._threadConnection then
--     tbl._threadConnection:send({ 'update', {[key] = value}  })
--   else
--     error("No parent or threadConnection!")
--   end
-- end

require 'nn'
require 'ParallelParallel'

pp = nn.ParallelParallel(1, 1)
for i = 1,1 do
  seq = nn.Sequential()
    seq:add(nn.Linear(10,9))
    seq:add(nn.Tanh())
    seq:add(nn.Reshape(3,3))
  pp:add(seq)
end

pp.modules[1].test = 0
