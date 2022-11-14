import numpy as np
import pygame
import skimage.measure
import matplotlib.pyplot as plt
from randomizeImage import * 

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def set_text(string, coordx, coordy, fontSize): #Function to set text

    font = pygame.font.Font('freesansbold.ttf', fontSize)
    text = font.render(string, True, WHITE) 
    textRect = text.get_rect()
    textRect.center = (coordx, coordy) 
    return (text, textRect)


def drawCanvas(network):
    pygame.init()


    answer = accuracy = None

    frames_per_second = 180
    draw_window_height = 364
    draw_window_width = 364
    total_window_width = 728
    total_window_height = 364
    text_window_width = total_window_width-draw_window_width


    blockXSize = int(draw_window_width/28)
    blockYSize = int(draw_window_height/28)


    # creating window
    display = pygame.display.set_mode((total_window_width, total_window_height))
    pygame.display.set_caption('Drawing canvas')
    # creating our frame regulator
    clock = pygame.time.Clock()

    def drawCircle(pos, color):
        pygame.draw.circle(display, color, pos, 15, 0)

    def erase_all():
        pygame.draw.rect(display, BLACK, (0, 0, draw_window_width, draw_window_height))
    
    def calculateImage():
        area = pygame.Rect(0, 0, draw_window_width, draw_window_height)
        sub_surface = display.subsurface(area)
        
        pixel_data = np.array(pygame.surfarray.array3d(sub_surface))
        pixel_data = skimage.measure.block_reduce(pixel_data, (1, 1, 3), np.max) #convert rgb in black/white
        pixel_data = pixel_data.reshape(pixel_data.shape[0], pixel_data.shape[1])
        
        unoriented_image = skimage.measure.block_reduce(pixel_data, (blockYSize, blockXSize), np.mean)
        oriented_image = np.zeros((28, 28))
        for row in range(unoriented_image.shape[1]):
            for col in range(unoriented_image.shape[0]):
                oriented_image[row][col] = unoriented_image[col][row]
        
        image = np.copy(oriented_image)
        image = image/255
        image[image < .1] = 0

        image = image.reshape(1, 28*28)
        return image

    def drawText(answer=None, accuracy=None):
        pygame.draw.rect(display, BLACK, (draw_window_width, 0, text_window_width, total_window_height)) #black surface
        totalText = set_text("Answer (Press enter to calculate):", int(total_window_width*3/4), 20, 20)
        display.blit(totalText[0], totalText[1])

        infoText = set_text("You can draw on the left side of", int(total_window_width*3/4), total_window_height-80, 15)
        display.blit(infoText[0], infoText[1])
        infoText = set_text("the screen by dragging the mouse", int(total_window_width*3/4), total_window_height-60, 15)
        display.blit(infoText[0], infoText[1])
        infoText = set_text("Then, press enter to calculate the NN answer,", int(total_window_width*3/4), total_window_height-40, 15)
        display.blit(infoText[0], infoText[1])
        infoText = set_text("and 'C' to erase your drawing and make a new one", int(total_window_width*3/4), total_window_height-20, 15)
        display.blit(infoText[0], infoText[1])

        if answer != None and accuracy != None:
            answerText = set_text("Answer: " + str(answer), int(total_window_width*3/4), 50, 20)
            display.blit(answerText[0], answerText[1])
            accuracyText = set_text("Accuracy: " + str(round(accuracy, 2)) + "%", int(total_window_width*3/4), 80, 20)
            display.blit(accuracyText[0], accuracyText[1])

    # forever loop
    flag = True
    
    display.fill(BLACK)
    drawText()
    while flag:

        clock.tick(frames_per_second)
        pygame.display.update()
        for event in pygame.event.get():
            keysPressed = pygame.key.get_pressed()
            if pygame.mouse.get_pressed()[0]: # If the current event is the mouse button down event
                pos = pygame.mouse.get_pos()
                drawCircle(pos, WHITE)

            elif pygame.mouse.get_pressed()[2]: # If the current event is the mouse button down event
                pos = pygame.mouse.get_pos()
                drawCircle(pos, BLACK)

            elif  keysPressed[pygame.K_c]: #if c is pressed, cancel everything 
                erase_all()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    image = calculateImage() #(1, 28*28)
                    answer, accuracy = network.run('selftest', image, [-1])
                    drawText(answer, accuracy)

            if event.type == pygame.QUIT:
                pygame.quit()
                flag = False
