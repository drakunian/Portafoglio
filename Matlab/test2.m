classdef DecisionTree
   properties
      nodes
      root
   end
   methods
      function obj = DecisionTree(root_node)
         obj.root = root_node;
         obj.nodes = {root_node};
      end
      
      function addnode(obj, parent_node, new_node)
          parent_node.children{end+1} = new_node;
          new_node.parent = parent_node;
          obj.nodes{end+1} = new_node;
      end

      function tree = createDecisionTree(data)
        % data: una tabella contenente le informazioni da inserire nei nodi dell'albero
        % restituisce: un oggetto Tree contenente l'albero decisionale
        
        % Creazione dell'albero decisionale
        tree = ('root');
        tree.Node{1} = 'Start';
        
        % Creazione dei nodi dell'albero
        for i=1:size(data, 1)
            nodeName = data.assets{i};
            parentNodeName = 'Start';
            nodeData = [data.weight(i), data.n_assets(i)];
            tree = tree.addnode(parentNodeName, nodeName, 'NodeData', nodeData);
        end

end

   end
end

