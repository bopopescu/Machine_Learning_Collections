im = imread('11ska580815.tif','tif') ;
axis image
figure(1)
image(im)

windowDiameter = 100 ;
windowRadius   = windowDiameter / 2 ;

stopCollecting = false ;
index = 72 ;
while ~stopCollecting
    [x,y] = ginput2(1,'KeepZoom') ;
    col = round(x) ;
    row = round(y) ;
    subimage = im(row - windowRadius:row + windowRadius,col - windowRadius:col + windowRadius,:) ;
    imwrite(subimage,sprintf('./image_fa_anything/image%g.tif',index),'tif')
    fprintf('Image %g\n',index)
    index = index + 1 ;
end

