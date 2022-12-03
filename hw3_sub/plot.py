import json
from subprocess import check_output, CalledProcessError, STDOUT, run
from pathlib import Path
from matplotlib import pyplot as plt
from collections import defaultdict

# plot the figure
def ploting(log):
    plt_name = 'Learning_Curves'
    for eval_type, values in log.items():
        x = list(range(1, len(values) + 1))
        plt.plot(x, values, label=eval_type)
    plt.xlabel('Batch500/step')
    plt.title(plt_name)
    plt.legend()
    plt.savefig('image.png')
    plt.close()


# save the ploting curve
def saving_curve():
    test_path = Path('data/public.jsonl')
    base_dir = Path('ckpt')
    base_name = 'checkpoint-'
    steps = sorted([int(str(filename).split('-')[-1])
                    for filename in base_dir.iterdir() if filename.name.startswith(base_name)])
    log = defaultdict(list)
    for s in steps:
        predict_path = Path('predictions') / f'{base_name}{s}.jsonl'
        predict_script = ['python', 'run_summarization_no_trainer.py', '--model_name_or_path',
                          base_dir /
                          f'{base_name}{s}', '--text_column', 'maintext',
                          '--validation_file', str(
                              test_path), '--output_path', predict_path,
                          '--num_beams', '8']
        run(predict_script)
        eval_script = ['python', 'eval.py', '-r',
                       str(test_path), '-s', str(predict_path)]
        result = check_output(eval_script)
        if isinstance(result, bytes):
            result = result.decode('utf-8')
        result = json.loads(result)
        for key, value in result.items():
            log[key].append(value.get('f'))
    ploting(log)


def print_it():
    base_dir = Path('predictions')
    for filename in base_dir.iterdir():
        script = ['python', 'eval.py', '-r',
                  'data/public.jsonl', '-s', filename]
        output = check_output(script)
        if isinstance(output, bytes):
            output = output.decode('utf-8')
        print('-' * 50)
        print(filename)
        print(output)
        print('-' * 50)



if __name__ == '__main__':
    saving_curve()
