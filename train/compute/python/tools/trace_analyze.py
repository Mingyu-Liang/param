import json
import matplotlib.pyplot as plt
import statistics


def read_dictionary_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data


def analyze_iteration(trace_events):
    cpu_ops = [event for event in trace_events if 'cat' in event and (event['cat'] == 'cpu_op' or event['cat'] == 'user_annotation')]
    cpu_ops = sorted(cpu_ops, key=lambda kv: kv['ts'])

    segments = {}
    segment = []
    last_segment_name = ''

    for op in cpu_ops:
        if 'ProfilerStep' in op['name']:
            if last_segment_name != '':
                segments[last_segment_name] = segment
            last_segment_name = op['name']
            segment = []
        else:
            segment.append(op)

    segments[last_segment_name] = segment

    segments_duration = {}
    for key, ops in segments.items():
        start_ts = ops[0]['ts']
        end_ts = ops[0]['ts'] + ops[0]['dur']
        for op in ops[1:]:
            start_ts = min(start_ts, op['ts'])
            end_ts = max(end_ts, op['ts'] + op['dur'])
        segments_duration[key] = (end_ts - start_ts)/ 1000.0

    # print(segments_duration)
    print(f'Avg: {statistics.mean(segments_duration.values())}, std: {statistics.stdev(segments_duration.values())}, max: {max(segments_duration.values())}, min: {min(segments_duration.values())}')


def analyze_nccl(trace_events):
    nccl_kernels = [event for event in trace_events if 'cat' in event and event['cat'] == 'kernel' and 'name' in event and event['name'].startswith('nccl')]
    nccl_kernels_durations = {}
    for kernel in nccl_kernels:
        if kernel['name'] not in nccl_kernels_durations:
            nccl_kernels_durations[kernel['name']] = []
        nccl_kernels_durations[kernel['name']].append(kernel['dur'] / 1000.0)

    # print(nccl_kernels_durations.keys())
    for key, value in nccl_kernels_durations.items():
        print(key, ' avg: ', statistics.mean(value), ' std: ', statistics.stdev(value), ' max: ', max(value), ' min: ', min(value))


def analyze_cpu_ops_duration(trace_events):
    cpu_ops = {}

    event_categories = set()
    for event in trace_events:
        if 'cat' not in event:
            continue
        event_categories.add(event['cat'])
        if event['cat'] == 'cpu_op':
            if event['name'] not in cpu_ops:
                cpu_ops[event['name']] = {}
                cpu_ops[event['name']]['duration'] = []
                cpu_ops[event['name']]['duration'].append(event['dur'])
            else:
                cpu_ops[event['name']]['duration'].append(event['dur'])    

    print(event_categories)
    print(cpu_ops.keys())

    cpu_ops_durations = {}
    for cpu_op in cpu_ops:
        cpu_ops_durations[cpu_op] = sum(cpu_ops[cpu_op]['duration'])

    cpu_ops_duration_total = sum(cpu_ops_durations.values())
    sorted_cpu_ops_durations = sorted(cpu_ops_durations.items(), key=lambda kv: kv[1], reverse=True)
    for cpu_op in sorted_cpu_ops_durations:
        print(cpu_op, ' fraction: ', cpu_op[1]/cpu_ops_duration_total)

    top10_cpu_ops = sorted_cpu_ops_durations[:10]
    print('top 10 cpu ops duration fraction: ', sum([cpu_op[1] for cpu_op in top10_cpu_ops])/cpu_ops_duration_total)
    top20_cpu_ops = sorted_cpu_ops_durations[:20]
    print('top 20 cpu ops duration fraction: ', sum([cpu_op[1] for cpu_op in top20_cpu_ops])/cpu_ops_duration_total)


def analyze_top_cpu_ops(trace_events):
    sorted_trace_events = sorted(trace_events, key=lambda kv: kv['ts'])

    cpu_events = [event for event in sorted_trace_events if 'cat' in event and event['cat'] == 'cpu_op']
    top_cpu_events = [cpu_events[0]]
    for event in cpu_events[1:]:
        if event['ts'] < top_cpu_events[-1]['ts'] + top_cpu_events[-1]['dur']:
            continue
        else:
            top_cpu_events.append(event)
    
    top_cpu_ops = {}
    for event in top_cpu_events:
        if event['name'] not in top_cpu_ops:
            top_cpu_ops[event['name']] = {}
            top_cpu_ops[event['name']]['duration'] = []
            top_cpu_ops[event['name']]['duration'].append(event['dur'])
        else:
            top_cpu_ops[event['name']]['duration'].append(event['dur'])    

    top_cpu_ops_durations = {}
    for cpu_op in top_cpu_ops:
        top_cpu_ops_durations[cpu_op] = sum(top_cpu_ops[cpu_op]['duration'])

    top_cpu_ops_duration_total = sum(top_cpu_ops_durations.values())
    sorted_top_cpu_ops_durations = sorted(top_cpu_ops_durations.items(), key=lambda kv: kv[1], reverse=True)
    
    for cpu_op in sorted_top_cpu_ops_durations:
        print(cpu_op, ' fraction: ', cpu_op[1]/top_cpu_ops_duration_total, ' avg: ', cpu_op[1] / len(top_cpu_ops[cpu_op[0]]['duration']) / 1000.0)

    print('Total cpu ops duration: ', top_cpu_ops_duration_total)
    top10_top_cpu_ops = sorted_top_cpu_ops_durations[:10]
    print('top 10 top cpu ops duration fraction: ', sum([cpu_op[1] for cpu_op in top10_top_cpu_ops])/top_cpu_ops_duration_total)
    top20_top_cpu_ops = sorted_top_cpu_ops_durations[:20]
    print('top 20 top cpu ops duration fraction: ', sum([cpu_op[1] for cpu_op in top20_top_cpu_ops])/top_cpu_ops_duration_total)

        
def analyze_cpu_ops_duration_under_specific_iteration(trace_events, iteration_name):
    iteration = [event for event in trace_events if 'name' in event and event['name'] == iteration_name]
    assert(len(iteration) == 1)
    sorted_trace_events = sorted(trace_events, key=lambda kv: kv['ts'])
    selected_trace_events = [event for event in sorted_trace_events if event['ts'] > iteration[0]['ts'] and event['ts'] < iteration[0]['ts'] + iteration[0]['dur']]
    
    cpu_events = [event for event in selected_trace_events if 'cat' in event and event['cat'] == 'cpu_op']
    cpu_ops = {}

    for event in cpu_events:
        if event['name'] not in cpu_ops:
            cpu_ops[event['name']] = {}
            cpu_ops[event['name']]['duration'] = []
            cpu_ops[event['name']]['duration'].append(event['dur'])
        else:
            cpu_ops[event['name']]['duration'].append(event['dur'])    

    cpu_ops_durations = {}
    for cpu_op in cpu_ops:
        cpu_ops_durations[cpu_op] = sum(cpu_ops[cpu_op]['duration'])

    cpu_ops_duration_total = sum(cpu_ops_durations.values())
    sorted_cpu_ops_durations = sorted(cpu_ops_durations.items(), key=lambda kv: kv[1], reverse=True)

    # for cpu_op in sorted_cpu_ops_durations:
    #     print(cpu_op, ' fraction: ', cpu_op[1]/cpu_ops_duration_total)

    top10_cpu_ops = sorted_cpu_ops_durations[:10]
    print(f'top 10 cpu ops duration fraction under {iteration_name}: ', sum([cpu_op[1] for cpu_op in top10_cpu_ops])/cpu_ops_duration_total)
    top20_cpu_ops = sorted_cpu_ops_durations[:20]
    print(f'top 20 cpu ops duration fraction under {iteration_name}: ', sum([cpu_op[1] for cpu_op in top20_cpu_ops])/cpu_ops_duration_total)


def analyze_top_cpu_ops_under_specific_iteration(trace_events, iteration_name):
    iteration = [event for event in trace_events if 'name' in event and event['name'] == iteration_name]
    assert(len(iteration) == 1)
    sorted_trace_events = sorted(trace_events, key=lambda kv: kv['ts'])

    cpu_events = [event for event in sorted_trace_events if 'cat' in event and event['cat'] == 'cpu_op']
    top_cpu_events = [cpu_events[0]]
    for event in cpu_events[1:]:
        if event['ts'] < top_cpu_events[-1]['ts'] + top_cpu_events[-1]['dur']:
            continue
        else:
            top_cpu_events.append(event)

    top_cpu_events_under_iteration = [event for event in top_cpu_events if event['ts'] > iteration[0]['ts'] and event['ts'] < iteration[0]['ts'] + iteration[0]['dur']]
    top_cpu_events_under_iteration_total_duration = sum([event['dur'] for event in top_cpu_events_under_iteration])

    aten_events = [event for event in sorted_trace_events if 'cat' in event and event['cat'] == 'cpu_op' and event['name'].startswith('aten')]
    top_aten_events = [aten_events[0]]
    for event in aten_events[1:]:
        if event['ts'] < top_aten_events[-1]['ts'] + top_aten_events[-1]['dur']:
            continue
        else:
            top_aten_events.append(event)
    
    top_aten_events_under_iteration = [event for event in top_aten_events if event['ts'] > iteration[0]['ts'] and event['ts'] < iteration[0]['ts'] + iteration[0]['dur']]
    top_aten_events_under_iteration_total_duration = sum([event['dur'] for event in top_aten_events_under_iteration])

    print(f'top cpu events under {iteration_name} total duration: ', top_cpu_events_under_iteration_total_duration,
          f'top aten events under {iteration_name} total duration: ', top_aten_events_under_iteration_total_duration,
        f', {iteration_name} total duration: ', iteration[0]['dur'], 
        ', cpu events fraction: ', top_cpu_events_under_iteration_total_duration / iteration[0]['dur'],
        ', aten events fraction: ', top_aten_events_under_iteration_total_duration / iteration[0]['dur'])


def analyze_kernels_duration(trace_events):
    kernels = {}

    event_categories = set()
    for event in trace_events:
        if 'cat' not in event:
            continue
        event_categories.add(event['cat'])
        if event['cat'] == 'kernel':
            if event['name'] not in kernels:
                kernels[event['name']] = {}
                kernels[event['name']]['duration'] = []
                kernels[event['name']]['duration'].append(event['dur'])
            else:
                kernels[event['name']]['duration'].append(event['dur'])    

    print(event_categories)
    print(kernels.keys())

    kernels_durations = {}
    for kernel in kernels:
        kernels_durations[kernel] = sum(kernels[kernel]['duration'])

    kernels_duration_total = sum(kernels_durations.values())
    sorted_kernels_durations = sorted(kernels_durations.items(), key=lambda kv: kv[1], reverse=True)
    for kernel in sorted_kernels_durations:
        print(kernel, ' fraction: ', kernel[1]/kernels_duration_total)

    top10_cpu_ops = sorted_kernels_durations[:10]
    print('top 10 kernels duration fraction: ', sum([cpu_op[1] for cpu_op in top10_cpu_ops])/kernels_duration_total)
    top20_cpu_ops = sorted_kernels_durations[:20]
    print('top 20 kernels duration fraction: ', sum([cpu_op[1] for cpu_op in top20_cpu_ops])/kernels_duration_total)


# Example usage
file_path = '/zhang-x3/users/ml2585/eg_logs/resnet_dist_4_long/resnet_dist_trace_1.json'  # Replace with your JSON file path
# file_path = '/tmp/tmp_20230530_21b8d82_287757.json'
dictionary = read_dictionary_from_json(file_path)
print(dictionary.keys())

trace_events = dictionary['traceEvents']

analyze_iteration(trace_events)
analyze_nccl(trace_events)

analyze_top_cpu_ops(trace_events)

exit(1)

# analyze_cpu_ops_duration(trace_events)

# analyze_cpu_ops_duration_under_specific_iteration(trace_events, 'iteration#17')

# analyze_top_cpu_ops_under_specific_iteration(trace_events, 'iteration#17')

# analyze_kernels_duration(trace_events)

iteration = [event for event in trace_events if 'name' in event and event['name'] == 'iteration#17']
assert(len(iteration) == 1)
sorted_trace_events = sorted(trace_events, key=lambda kv: kv['ts'])

trace_events_under_iteration = [event for event in sorted_trace_events if event['ts'] > iteration[0]['ts'] and event['ts'] < iteration[0]['ts'] + iteration[0]['dur']]

print(len(trace_events_under_iteration))

cuda_runtime = [event for event in trace_events_under_iteration if event['cat'] == 'cuda_runtime']
# kernels = [event for event in trace_events_under_iteration if event['cat'] == 'kernel']
kernels = [event for event in trace_events_under_iteration if event['cat'] == 'kernel' or event['cat'] == 'gpu_memset' or event['cat'] == 'gpu_memcpy']

print(len(cuda_runtime), len(kernels))

external_id_cuda_kernel = {}

for cuda in cuda_runtime:
    if 'args' in cuda and 'External id' in cuda['args']:
        external_id = cuda['args']['External id']
        if external_id not in external_id_cuda_kernel:
            external_id_cuda_kernel[external_id] = {}
            external_id_cuda_kernel[external_id]['cuda'] = cuda['ts']
        else:
            external_id_cuda_kernel[external_id]['cuda'] = cuda['ts']

for kernel in kernels:
    if 'args' in kernel and 'External id' in kernel['args']:
        external_id = kernel['args']['External id']
        if external_id not in external_id_cuda_kernel:
            external_id_cuda_kernel[external_id] = {}
            external_id_cuda_kernel[external_id]['kernel'] = kernel['ts']
        else:
            external_id_cuda_kernel[external_id]['kernel'] = kernel['ts']

delays_ms = []
for external_id in external_id_cuda_kernel:
    if len(external_id_cuda_kernel[external_id]) != 2:
        print(external_id, external_id_cuda_kernel[external_id])
    else:
        delays_ms.append((external_id_cuda_kernel[external_id]['kernel'] - external_id_cuda_kernel[external_id]['cuda']) / 1000)

print('mean: ', statistics.mean(delays_ms), 'std: ', statistics.stdev(delays_ms))

# Create a histogram plot
plt.hist(delays_ms, bins=10, edgecolor='black')

# Set the plot labels and title
plt.xlabel('Delay')
plt.ylabel('Frequency')
plt.title('Distribution of Delays')

# Save the plot to a file
plt.savefig('delays_histogram.png')
