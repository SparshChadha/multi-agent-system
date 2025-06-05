import graphviz

# Create a directed graph
dot = graphviz.Digraph(comment='Multi-Format Autonomous AI System')

# Add nodes
dot.node('Classifier', 'Classifier')
dot.node('Router', 'Router')
dot.node('EmailAgent', 'EmailAgent')
dot.node('JSONAgent', 'JSONAgent')
dot.node('PDFAgent', 'PDFAgent')
dot.node('ActionRouter', 'ActionRouter')

# Add edges
dot.edge('Classifier', 'Router')
dot.edge('Router', 'EmailAgent', label='agent = Email')
dot.edge('Router', 'JSONAgent', label='agent = JSON')
dot.edge('Router', 'PDFAgent', label='agent = PDF')
dot.edge('EmailAgent', 'ActionRouter')
dot.edge('JSONAgent', 'ActionRouter')
dot.edge('PDFAgent', 'ActionRouter')

# Render and display
dot.format = 'png'
dot.render('graph_architecture', view=False)

# Display the graph
from IPython.display import Image
Image(filename='graph_architecture.png')
