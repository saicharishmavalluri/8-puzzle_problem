#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime
import sys


# In[2]:


def check_solvable(state):
    """Checks if the given state is solvable by checking the inversion pairs 
    and returns true or false based on the result"""
    count = 0
    blank = '_'
    list_version = [elem for row in state for elem in row]
    for i in range(0, 9):
        for j in range(i + 1, 9):
            if (list_version[j] != blank and list_version[i] != blank):
                if(list_version[i] > list_version[j]):
                    count += 1
    return (count % 2 == 0)


# In[14]:


class Node():
    """Node class to store the parent, action, depth of the current node"""
    def __init__(self, state, action, parent, depth):
        self.state = state;
        self.action = action;
        self.parent = parent;
        self.depth = depth;
    
    def check_action_left(self):
        """This function checks if the action to move '_' 
        to the left is valid and returns true/false based on the result"""
        index_loc=[i[0] for i in np.where(self.state=='_')]
        if index_loc[1] in [0,3,6]:
            return False
        else:
            element = self.state[index_loc[0],index_loc[1]-1]
            new_state = self.state.copy()
            new_state[index_loc[0],index_loc[1]] = element
            new_state[index_loc[0],index_loc[1]-1] = '_'
            return new_state;
        
    def check_action_right(self):
        """This function checks if the action to move '_' 
        to the right is valid and returns true/false based on the result"""
        index_loc=[i[0] for i in np.where(self.state=='_')]
        if index_loc[1] in [2,5,8]:
            return False
        else:
            element = self.state[index_loc[0],index_loc[1]+1]
            new_state = self.state.copy()
            new_state[index_loc[0],index_loc[1]] = element
            new_state[index_loc[0],index_loc[1]+1] = '_'
            return new_state;
        
    def check_action_up(self):
        """This function checks if the action to move '_' 
        to the up is valid and returns true/false based on the result"""
        index_loc=[i[0] for i in np.where(self.state=='_')]
        if index_loc[0] == 0:
            return False
        else:
            element = self.state[index_loc[0]-1,index_loc[1]]
            new_state = self.state.copy()
            new_state[index_loc[0],index_loc[1]] = element
            new_state[index_loc[0]-1,index_loc[1]] = '_'
            return new_state;
        
    def check_action_down(self):
        """This function checks if the action to move '_' 
        to the down is valid and returns true/false based on the result"""
        index_loc=[i[0] for i in np.where(self.state=='_')]
        if index_loc[0] == 2:
            return False
        else:
            element = self.state[index_loc[0]+1,index_loc[1]]
            new_state = self.state.copy()
            new_state[index_loc[0],index_loc[1]] = element
            new_state[index_loc[0]+1,index_loc[1]] = '_'
            return new_state;
 
    def misplaced_tiles(self,new_state,goal_state):
        """Checks how many tiles are misplaced and 
        returns the number of misplaced tiles if present otherwise returns 0"""
        cost=0
        for i in range(3):
            for j in range(3):
                if(new_state[i][j] != '_'):
                    if(new_state[i][j] != goal_state[i][j]):
                        cost=cost+1
        return cost;
    
    def manhattan_distance(self,new_state,goal_state):
        """Heuristic function to return the sum of manhattam distances for a given state"""
        goal_state_pos = {'1':(0,0),'2':(0,1),'3':(0,2),
                          '4':(1,0),'5':(1,1),'6':(1,2),
                          '7':(2,0),'8':(2,1),'_':(2,2)} 
        manhattan = 0
        for i in range(3):
            for j in range(3):
                if new_state[i,j] != '_':
                    zip_object = zip((i,j), goal_state_pos[new_state[i,j]])
                    manhattan += sum(abs(a-b) for a,b in zip_object)
        return manhattan
    
    
    def outofrow_outofcolumn(self,new_state,goal_state):
        """h3 is an admissible heuristic, since every tile that is out of column or 
    out of row must be moved at least once and every tile that is 
    both out of column and out of row must be moved at least twice."""
        goal_state_pos = {'1':(0,0),'2':(0,1),'3':(0,2),
                          '4':(1,0),'5':(1,1),'6':(1,2),
                          '7':(2,0),'8':(2,1),'_':(2,2)} 
        row_count = 0
        column_count = 0
        for i in range(3):
            for j in range(3):
                if new_state[i,j] != '_':
                    if new_state[i,j] != goal_state[i,j]:
                        row_count = row_count + abs(i - goal_state_pos[new_state[i,j]][0])
                        column_count = column_count + abs(j - goal_state_pos[new_state[i,j]][1])
    
        return (row_count+column_count)
    
    
    def heuristic_cost(self,new_state,goal_state,heuristic_function,path_cost,depth):
        """a function that plugs in the respective to the heuristic cost"""
        if heuristic_function == 'h1':
            return self.misplaced_tiles(new_state,goal_state)
        elif heuristic_function == 'h2':
            return self.manhattan_distance(new_state,goal_state)
        # since this game is made unfair by setting the step cost as the value of the tile being moved
        # to make it fair, I made all the step cost as 1
        # made it a best-first-search with manhattan heuristic function
        elif heuristic_function == 'h3':
            return self.outofrow_outofcolumn(new_state, goal_state)
        
    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))
        
        
    def Breadth_First_Search(self, goal_state):
        start_time = datetime.datetime.now()
        
        queue = [self]
        queue_length = 1
        explored = set([])
        str_sol = "";
        path_count=0
    
        while(queue):
            present_time = datetime.datetime.now()
            difference = present_time-start_time
            minutes = divmod(difference.total_seconds(),60)[0]
            if(minutes >= 15.0):
                print('Total nodes generated: <<??>>')
                print('Total time taken: >= 15 min')
                print('Path Length: Timed out.')
                print('Path: Timed out.')
                break;
            if(len(queue) > queue_length):
                queue_length = len(queue)
            current_node = queue.pop(0)
            explored.add(tuple(current_node.state.reshape([1,9])[0]))
            if(np.array_equal(current_node.state,goal_state)):
                print('Total nodes generated: '+str(queue_length))
                present_time = datetime.datetime.now()
                difference = present_time-start_time
                seconds = difference.total_seconds()
                print('Total time taken: '+str(seconds)+'sec.')
                list_sol = current_node.solution();
                for i in list_sol:
                    str_sol = str_sol+str(i)
                    path_count = path_count+1
                print("Path length: "+str(path_count))
                print("Path : "+str_sol)
                return True;
                
            else:
                if current_node.check_action_left() is not False:
                    result_state = current_node.check_action_left()
                    if(tuple(result_state.reshape([1,9])[0]) not in explored):
                        new_node = Node(state = result_state, action = 'R', parent = current_node, depth = None)
                        queue.append(new_node)
                        
                       
                if current_node.check_action_right() is not False:
                    result_state = current_node.check_action_right()
                    if(tuple(result_state.reshape([1,9])[0]) not in explored):
                        new_node = Node(state = result_state, action = 'L', parent = current_node,depth = None)
                        queue.append(new_node)
                        
                
                if current_node.check_action_up() is not False:
                    result_state = current_node.check_action_up()
                    if(tuple(result_state.reshape([1,9])[0]) not in explored):
                        new_node = Node(state = result_state, action = 'D', parent = current_node,depth = None)
                        queue.append(new_node)
                        
               
                if current_node.check_action_down() is not False:
                    result_state = current_node.check_action_down()
                    if(tuple(result_state.reshape([1,9])[0]) not in explored):
                        new_node = Node(state = result_state, action = 'U', parent = current_node,depth = None)
                        queue.append(new_node)
                
                
    def Iterative_Deepening_Search(self, goal_state):
        start_time = datetime.datetime.now()
        queue_length = 1
        for depth_limit in range(sys.maxsize):
            queue = [self]
            depth_queue = [0]
            explored = set([])
            str_sol = "";
            path_count=0
        
            while(queue):
                present_time = datetime.datetime.now()
                difference = present_time-start_time
                minutes = divmod(difference.total_seconds(),60)[0]
                if(minutes >= 15.0):
                    print('Total nodes generated: <<??>>')
                    print('Total time taken: >= 15 min')
                    print('Path Length: Timed out.')
                    print('Path: Timed out.')
                    break;
                if(len(queue) > queue_length):
                    queue_length = len(queue)
                current_index = depth_queue.index(max(depth_queue));
                current_depth = depth_queue.pop(current_index);
                current_node = queue.pop(current_index);
                explored.add(tuple(current_node.state.reshape([1,9])[0]))
                if(np.array_equal(current_node.state,goal_state)):
                    print('Total nodes generated: '+str(queue_length))
                    present_time = datetime.datetime.now()
                    difference = present_time-start_time
                    seconds = difference.total_seconds()
                    print('Total time taken: '+str(seconds)+'sec.')
                    list_sol = current_node.solution();
                    for i in list_sol:
                        str_sol = str_sol+str(i)
                        path_count = path_count+1
                    print("Path length: "+str(path_count))
                    print("Path : "+str_sol)
                    return True;

                else:
                    if current_depth < depth_limit:
                        if current_node.check_action_left() is not False:
                            result_state = current_node.check_action_left()
                            if(tuple(result_state.reshape([1,9])[0]) not in explored):
                                new_node = Node(state = result_state, action = 'R', parent = current_node, depth = current_depth)
                                queue.append(new_node)
                                depth_queue.append(current_depth+1)
                                


                        if current_node.check_action_right() is not False:
                            result_state = current_node.check_action_right()
                            if(tuple(result_state.reshape([1,9])[0]) not in explored):
                                new_node = Node(state = result_state, action = 'L', parent = current_node,depth = current_depth)
                                queue.append(new_node)
                                depth_queue.append(current_depth+1)
                                


                        if current_node.check_action_up() is not False:
                            result_state = current_node.check_action_up()
                            if(tuple(result_state.reshape([1,9])[0]) not in explored):
                                new_node = Node(state = result_state, action = 'D', parent = current_node,depth = current_depth)
                                queue.append(new_node)
                                depth_queue.append(current_depth+1)
                                


                        if current_node.check_action_down() is not False:
                            result_state = current_node.check_action_down()
                            if(tuple(result_state.reshape([1,9])[0]) not in explored):
                                new_node = Node(state = result_state, action = 'U', parent = current_node,depth = current_depth)
                                queue.append(new_node)
                                depth_queue.append(current_depth+1)
                            
        
    def a_star_search(self, goal_state, h_function):
        start_time = datetime.datetime.now()
        
        queue = [(self,0)]
        depth_queue = [(0,0)]
        path_queue = [(0,0)]
        queue_length = 1
        explored = set([])
        str_sol = "";
        path_count=0
        total_cost = 0
        h_cost = 0
    
        while(queue):
            queue = sorted(queue, key=lambda x: x[1])
            depth_queue = sorted(depth_queue, key=lambda x: x[1])
            path_queue = sorted(path_queue, key=lambda x: x[1])
            present_time = datetime.datetime.now()
            difference = present_time-start_time
            minutes = divmod(difference.total_seconds(),60)[0]
            if(minutes >= 15.0):
                print('Total nodes generated: <<??>>')
                print('Total time taken: >= 15 min')
                print('Path Length: Timed out.')
                print('Path: Timed out.')
                break;
            if(len(queue) > queue_length):
                queue_length = len(queue)
            current_node = queue.pop(0)[0];
            current_depth = depth_queue.pop(0)[0];
            current_path_cost = path_queue.pop(0)[0];
            explored.add(tuple(current_node.state.reshape([1,9])[0]))
            if(np.array_equal(current_node.state,goal_state)):
                print('Total nodes generated: '+str(queue_length))
                present_time = datetime.datetime.now()
                difference = present_time-start_time
                seconds = difference.total_seconds()
                print('Total time taken: '+str(seconds)+'sec.')
                list_sol = current_node.solution();
                for i in list_sol:
                    str_sol = str_sol+str(i)
                    path_count = path_count+1
                print("Path length: "+str(path_count))
                print("Path : "+str_sol)
                return True;
                
            else:
                if current_node.check_action_left() is not False:
                    result_state = current_node.check_action_left()
                    if(tuple(result_state.reshape([1,9])[0]) not in explored):
                        h_cost = self.heuristic_cost(result_state, goal_state, h_function,current_path_cost, current_depth)
                        total_cost = current_path_cost+h_cost
                        new_node = Node(state = result_state, action = 'R', parent = current_node, depth = current_depth)
                        queue.append((new_node,total_cost))
                        depth_queue.append((current_depth+1,total_cost))
                        path_queue.append((current_path_cost+1,total_cost))


                if current_node.check_action_right() is not False:
                    result_state = current_node.check_action_right()
                    if(tuple(result_state.reshape([1,9])[0]) not in explored):
                        h_cost = self.heuristic_cost(result_state, goal_state, h_function,current_path_cost, current_depth)
                        total_cost = current_path_cost+h_cost
                        new_node = Node(state = result_state, action = 'L', parent = current_node,depth = current_depth)
                        queue.append((new_node,total_cost))
                        depth_queue.append((current_depth+1,total_cost))
                        path_queue.append((current_path_cost+1,total_cost))


                if current_node.check_action_up() is not False:
                    result_state = current_node.check_action_up()
                    if(tuple(result_state.reshape([1,9])[0]) not in explored):
                        h_cost = self.heuristic_cost(result_state, goal_state, h_function,current_path_cost, current_depth)
                        total_cost = current_path_cost+h_cost
                        new_node = Node(state = result_state, action = 'D', parent = current_node,depth = current_depth)
                        queue.append((new_node,total_cost))
                        depth_queue.append((current_depth+1,total_cost))
                        path_queue.append((current_path_cost+1,total_cost))

                if current_node.check_action_down() is not False:
                    result_state = current_node.check_action_down()
                    if(tuple(result_state.reshape([1,9])[0]) not in explored):
                        h_cost = self.heuristic_cost(result_state, goal_state, h_function,current_path_cost, current_depth)
                        total_cost = current_path_cost+h_cost
                        new_node = Node(state = result_state, action = 'U', parent = current_node,depth = current_depth)
                        queue.append((new_node,total_cost))
                        depth_queue.append((current_depth+1,total_cost))
                        path_queue.append((current_path_cost+1,total_cost))
                              


# In[13]:


file_path = input("Enter the file path")
algorithm = input("Enter the algorithm that needed to be used (BFS/IDS/h1/h2/h3)")
print("Path given: "+file_path);
print("Algorithm given: "+algorithm);
goal_state = np.array(['1','2','3','4','5','6','7','8','_']).reshape(3,3)
input_file = pd.read_csv(file_path, header=None);

#converting the dataframe input_file into numpy for better comparision
numpy_array = input_file.to_numpy()
List_puzzle = []
for i in range(0,len(numpy_array)):
    index_array = numpy_array[i][0]
    List_puzzle = List_puzzle+index_array.split(' ')
initial_state = np.array(List_puzzle).reshape(3,3)

if(not check_solvable(initial_state)):
    print('The inputted puzzle is not solvable:')
    print(initial_state)
else:
    root_node = Node(state = initial_state, action = None, parent = None, depth = None)
    if(algorithm == 'BFS'):
        root_node.Breadth_First_Search(goal_state)
    elif(algorithm == 'IDS'):
        root_node.Iterative_Deepening_Search(goal_state)
    elif(algorithm in ['h1','h2','h3']):
        root_node.a_star_search(goal_state,h_function = algorithm)
    else:
        print("Incorrect algorithm input")
        print("Please enter BFS or IDS or h1 or h2 or h3")
        


# In[ ]:




