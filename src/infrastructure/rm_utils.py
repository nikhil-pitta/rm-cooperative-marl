import os

def build_rm(file_paths):
    # Extract folder name from the first file path
    folder_name = os.path.dirname(file_paths[0])
    combined_file_name = os.path.join(folder_name, os.path.basename(folder_name) + '_rm_joined.txt')
    set_list = []

    state_offset = 1
    aux_connections = []
    with open(combined_file_name, 'w') as combined_file:
        # Write the auxiliary node
        combined_file.write('0 # auxiliary node\n')

        for file_path in file_paths:
            states_set = set()
            with open(file_path, 'r') as file:
                for line in file:
                    # Skip the initial state line as it will be connected to the auxiliary node
                    if line.strip() == '0 # initial state':
                        aux_connections.append(f'(0, {state_offset}, "to_rm{len(aux_connections)+1}", 0) # Connect to RM {len(aux_connections)+1}\n')
                        continue

                    # Modify state numbers and write to the combined file
                    if line.startswith('('):
                        line = line.split("#")
                        comment = line[1].strip()
                        line = line[0]
                        parts = line.strip()[1:-1].split(',')
                        parts[0] = str(int(parts[0].strip()) + state_offset)
                        states_set.add(int(parts[0].strip()))
                        parts[1] = str(int(parts[1].strip()) + state_offset)
                        states_set.add(int(parts[1].strip()))
                        combined_file.write('(' + ', '.join(parts) + ') ' + "#" + comment + '\n' )
                    else:
                        combined_file.write(line)
                set_list.append(states_set)

            # Update the offset for the next RM
            state_offset += len(states_set)

        # Write connections from auxiliary node to initial states of each RM
        for connection in aux_connections:
            combined_file.write(connection)
    return combined_file_name, set_list