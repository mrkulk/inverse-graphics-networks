require 'parallel'

function intermediate()
  -- function child()

  --   while true do
  --     parallel.yield()
  --     parallel.print("I'm a child!")
  --     parallel.parent:send("next")
  --   end
  -- end
  for i = 1, parallel.nchildren do
    parallel.children[i]:join()
  end
  print(parallel.children:receive())

end

function parent()
  local numChildren = 10
  parallel.nfork(numChildren)
  parallel.children:exec(intermediate)
end

function setup()

  function child()
    while true do
      parallel.yield()
      parallel.print("I'm a child!")
      parallel.parent:send("next")
    end
  end

  parallel.nfork(4)
  parallel.children:exec(child)
end

setup()
for i = 1,30 do

  intermediate()
end



-- ok, err = pcall(parent)
-- if not ok then print(err) parallel.close() end























