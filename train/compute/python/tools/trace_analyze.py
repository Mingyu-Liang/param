import json

def read_dictionary_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data


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
file_path = '/zhang-x3/users/ml2585/eg_logs/resnet_trace_new.json'  # Replace with your JSON file path
# file_path = '/tmp/tmp_20230530_21b8d82_287757.json'
dictionary = read_dictionary_from_json(file_path)
print(dictionary.keys())

trace_events = dictionary['traceEvents']

# analyze_cpu_ops_duration(trace_events)

# analyze_top_cpu_ops_under_specific_iteration(trace_events, 'iteration#17')

analyze_kernels_duration(trace_events)

