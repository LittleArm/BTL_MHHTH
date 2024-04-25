import numpy as np
import networkx as nx
import pandas as pd
# import matplotlib.pyplot as plt
def min_cost_flow(G, source, sink, flow, stop_time, demand_info, stage, copy):## demand = 1 in ra nội dung node dừng, demand = 0 in chi phí
    # Khởi tạo các biến
    n = len(G.nodes)  # Số lượng nút trong đồ thị
    total_flow = 0  # Tổng dòng chảy hiện tại
    save_node = []
    # Vòng lặp chính của thuật toán Bellman-Ford
    while total_flow < flow:  # Bước 2: Dừng lại khi dòng chảy đạt giá trị mong muốn
        dist = [float('inf')] * n  # Mảng khoảng cách từ nguồn đến mỗi nút
        prev_node = [None] * n  # Mảng lưu nút trước đó trên đường đi ngắn nhất từ nguồn
        dist[source] = 0  # Khoảng cách từ nguồn đến chính nó là 0
        for _ in range(n):
            for u, v, data in G.edges(data=True):
                w = data['capacity']
                if stage == 2 and copy == 2:
                    cost = data['weight']  # Thêm penalty vào chi phí của cạnh
                elif stage == 1 or (stage == 2 and copy == 3):
                    cost = data['penalty']
                if w > 0 and dist[u] + cost < dist[v]:
                    dist[v] = dist[u] + cost
                    if stage == 2 and copy == 3:
                        prev_node[v] = (u,v,w, data['weight'])
                    else:
                        prev_node[v] = (u, v, w, cost)

        #Kiểm tra xem có còn đường đi từ nguồn đến đích không
        if prev_node[sink] is None:
            print("Không có đường đi")
            break

        # Tìm đường đi ngắn nhất từ nguồn đến đích
        node = sink
        path = []
        while node != source:
            path.append(prev_node[node])
            node = prev_node[node][0]
        path = path[::-1]

        # Tính toán dòng chảy nhỏ nhất trên đường đi
        min_flow = min(w for _, _, w, _ in path)
        min_flow = min(min_flow, flow - total_flow)  # Không cho phép dòng chảy vượt quá giá trị mong muốn

        save_node.append(print_flow(path, min_flow, stop_time,demand_info))
        # Cập nhật công suất của các cạnh trên đường đi
        for u, v, w, cost in path:
            G[u][v]['capacity'] -= min_flow
            if u in G[v]:
                G[v][u]['capacity'] += min_flow

        total_flow += min_flow  # Bước 3: Tăng dòng chảy

    # Trả về chi phí của dòng chảy
    if demand_info == 0:
        TIME = 0
        for a,b,c in save_node:
            TIME += b
        return TIME
    if demand_info == 1:
        return save_node

def print_flow(path, min_flow, stop_time, demand_info): #hàm in thông tin đường đi
    time = 0
    node_stop = 0
    for u,v,w,cost in path:
        print(f"Path: Node {u} -> Node {v}, Flow = {min_flow}")
        time += cost * min_flow
        if time > stop_time and demand_info == 1:
            node_stop = u
            break
        print(f"cost:{time}")
    if node_stop > 0:
        print(f"{min_flow} dừng tại node {node_stop}")
    print("---------------------------------------")
    return (node_stop, time, min_flow)



# Số lượng nút và liên kết trong mạng lưới
num_nodes = int(input("Nhập số lượng nút thực tế: "))

# Số lượng xe cần được sơ tán
num_cars = int(input("Nhập tổng số lượng xe cần sơ tán tại tất cả nút nguồn: "))

# Danh sách các nút nguồn
num_sources = int(input(f"Nhập số lượng nút nguồn (<{num_nodes}): "))
sources_array = []
for i in range(num_sources):
    value = int(input(f"Nhập nhãn của nút nguồn {i + 1} (0<x<={num_nodes}): "))
    sources_array.append(value)

# Danh sách các nút đích
num_sinks = int(input(f"Nhập số lượng nút đích (<={num_nodes - num_sources}): "))
sinks_array = []
for i in range(num_sinks):
    value = int(input(f"Nhập nhãn của nút đích {i + 1} (0<x<={num_nodes} và khác nút nguồn): "))
    sinks_array.append(value)



# Thời điểm diễn ra thảm họa
time_limit = int(np.random.uniform(10, 400, 1)[0])
time_threshold = int(np.random.uniform(time_limit/4, 3 * time_limit/4, 1)[0]) # phút (thời gian chạy đúng theo kế hoạch)
print("Thời điểm xảy ra thảm họa: phút thứ ", time_threshold)

# Khởi tạo weighted adjacency matrix
col_begin = np.zeros((num_nodes,1))
row_end = np.zeros((1, num_nodes + 1))
row_begin = np.zeros((1, num_nodes + 1))
col_end = np.zeros((num_nodes + 2, 1))
net_capacity = None

# Tổng capacity từ supersource tới các sources bằng num_cars
if num_sources > 1:
    sum_cars = 0
    for i in range(num_sources - 1):
        row_begin[0, sources_array[i]] = int(np.random.uniform(1, num_cars/num_sources, 1)[0])
        sum_cars += row_begin[0, sources_array[i]]
    row_begin[0, sources_array[num_sources - 1]] = num_cars - sum_cars
else:
    row_begin[0, sources_array[0]] = num_cars

# Tổng capacity từ sinks đến supersink bằng num_cars
if num_sinks > 1:
    sum_cars = 0
    for i in range(num_sinks - 1):
        col_end[sinks_array[i], 0] = int(np.random.uniform(1, num_cars/num_sinks, 1)[0])
        sum_cars += col_end[sinks_array[i], 0]
    col_end[sinks_array[num_sinks - 1], 0] = num_cars - sum_cars
else:
    col_end[sinks_array[0], 0] = num_cars


def initial_capacity_matrix(num_nodes, num_cars):  # Tạo capacity_net ngẫu nhiên
    capacity = np.round(np.random.uniform(low=num_cars/10, high=num_cars/2, size=(num_nodes, num_nodes))).astype(int)
    np.fill_diagonal(capacity, 0)
    mask = np.random.choice([0, 1], size=capacity.shape, p=[0.7, 0.3])
    capacity *= mask

    # Thêm điều kiện: nếu capacity[i][j] > 0 thì capacity[j][i] = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if capacity[i][j] > 0:
                capacity[j][i] = 0

    return capacity


def has_path(adjacency_matrix, start_node, target_node, visited=None):  # Kiểm tra có path giữa supersource-supersink không
    if visited is None:
        visited = set()
    visited.add(start_node)
    if start_node == target_node:
        return True
    for neighbor, connected in enumerate(adjacency_matrix[start_node]):
        if connected != 0 and neighbor not in visited:
            if has_path(adjacency_matrix, neighbor, target_node, visited):
                return True
    return False



# Khởi tạo net_capacity, break khi mà có path giữa supersource-supersink
while True:
    net_capacity = initial_capacity_matrix(num_nodes, num_cars)
    net_capacity = np.hstack([col_begin, net_capacity])  # add new col to the left
    net_capacity = np.vstack([net_capacity, row_end])  # add new row to the bottom
    net_capacity = np.vstack([row_begin, net_capacity])  # add new row to the top
    net_capacity = np.hstack([net_capacity, col_end])  # add new col to the right
    if has_path(net_capacity, 0, num_nodes + 1):
        break

# Khởi tạo net_time
net_time = net_capacity.copy()
nonzero_indices = np.nonzero(net_time)
for i, j in zip(*nonzero_indices):
    net_time[i, j] = np.random.choice([1, 2, 3, 4, 5])
net_time[0, :] = 0
net_time[:, num_nodes + 1] = 0

# Khởi tạo net_penalty
net_penalty = net_capacity.copy()
nonzero_indices = np.nonzero(net_time)
for i, j in zip(*nonzero_indices):
    net_penalty[i, j] = np.random.choice([1, 2, 3, 4, 5])
net_penalty[0, :] = 0
net_penalty[:, num_nodes + 1] = 0

def export_excel(net, quantity):  # Xuất net ra excelS
    df = pd.DataFrame(net)
    if quantity == "Capacity":
        excel_path = 'C:/Users/Hi HP/Desktop/MM/Capacity.xlsx'
    else:
        excel_path = 'C:/Users/Hi HP/Desktop/MM/Time.xlsx'
    df.to_excel(excel_path, index=False, header=False)


export_excel(net_capacity, "Capacity")
export_excel(net_penalty, "Time")

def update_net(G, num_cars):
    for u, v, data in G.edges(data=True):
        # Cập nhật công suất (capacity) và thời gian di chuyển (time_travel) một cách ngẫu nhiên
        data['capacity'] = np.random.randint(num_cars/10, num_cars/2)  # Giả sử công suất mới nằm trong khoảng từ 1 đến 10
        data['time_travel'] = np.random.choice([1, 2, 3, 4, 5])  # Giả sử thời gian di chuyển mới nằm trong khoảng từ 0.1 đến 2.0

# Tui không biết có cần thay đổi time_threshold ở stage 2 không, nếu cần thì ông copy cái time_threshold ban đầu là nó random lại
# Khởi tạo mạng lưới
G = nx.DiGraph()
############################ kế hoạch ###################################### stage 1
# Thêm các nút và cung vào mạng lưới
for i in range(num_nodes + 2):
    for j in range(num_nodes + 2):
        if net_capacity[i][j] > 0:
            G.add_edge(i, j, capacity=net_capacity[i][j], weight=net_time[i][j], penalty = net_penalty[i][j])
            if net_capacity[j][i] == 0:
                G.add_edge(j, i, capacity=0, weight=net_time[i][j], penalty = net_penalty[i][j])  # Thêm cạnh ngược lại


# Tìm luồng có chi phí nhỏ nhất thỏa mãn tất cả các yêu cầu
print("--------------------------Stage 1-----------------------------")
flow_cost = min_cost_flow(G, 0,num_nodes + 1, num_cars, time_threshold, 0, 1, 2)
# in danh sách đường đi
print(f"giá trị min_cost của stage 1 là: {flow_cost} phút")
#reset lại mạng G
for i in range(num_nodes + 2):
    for j in range(num_nodes + 2):
        if net_capacity[i][j] > 0:
            G.add_edge(i, j, capacity=net_capacity[i][j], weight=net_time[i][j])
            if net_capacity[j][i] == 0:
                G.add_edge(j, i, capacity=0, weight=net_time[i][j])  # Thêm cạnh ngược lại
print(f"-------------------------Stage 2 tại thời điểm {time_threshold} phút -----------------------")
# Giai đoạn 2: Thiên tai đã bắt đầu
flow = min_cost_flow(G, 0, num_nodes + 1, num_cars, time_threshold,1, 2, 3)
# Giả sử rằng sau khi thiên tai bắt đầu, sức chứa của một số cung bị giảm đi
MIN_COST = 0
for k in range(10):
    print(f"------------------------------scenerio {k + 1}---------------------------")
    update_net(G,num_cars)
    # Giải quyết bài toán với sức chứa mới
    total_cost = 0
    for node_stop, time, flow_now in flow:
        if node_stop == 0:
            print(f"{flow_now} xe về an toàn trước khi thảm họa")
            continue
        print(f"{flow_now} xe tiếp tục đi tiếp tại node {node_stop}:")
        flowCost = min_cost_flow(G, node_stop, num_nodes + 1, flow_now,time_threshold, 0, 2, 2)
        total_cost += flowCost
    print(f"Giá trị min cost scenario {k + 1}: {total_cost} phút")
    MIN_COST += total_cost

MIN_COST = float(MIN_COST / 10)
TIME = 0
for a,b,c in flow:
    TIME += b
print(f"------------------------- Kết thúc ----------------------------------")
print(f"kết luận min_cost cần tìm là : {MIN_COST + TIME + flow_cost}")
