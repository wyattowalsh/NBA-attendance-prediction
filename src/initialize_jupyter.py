import multiprocessing
import warnings
from IPython.display import display, HTML

# Enable multiprocessing for use in model building
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

# Configure notebook display and warnings
display(HTML("<style>.container { width:100% !important; }</style>"))
# display(HTML("<style>div.output_scroll { height: 48em; }</style>"))
display(HTML("""<style> .output_png {display: table-cell;
        text-align: center;
        vertical-align: middle;
    }</style>"""))
# CSS = """
# div.cell:nth-child(3) .output {
#     flex-direction: row;
# }
# """
# HTML('<style>{}</style'.format(CSS))

	# warnings.filterwarnings('ignore')


