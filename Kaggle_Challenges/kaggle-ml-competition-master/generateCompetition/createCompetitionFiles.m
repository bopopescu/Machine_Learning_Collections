iFa = 1 ;
iTarget = 1 ;
maxFa = 180 ;
maxTarget = 150 ;

solution = nan(maxFa + maxTarget,2) ;

index = 1 ;
while (iFa <= maxFa) || (iTarget <= maxTarget)
    rVal = rand < (maxTarget / (maxFa + maxTarget)) ;
    
    if (iFa <= maxFa) && ~rVal
        originalName = sprintf('./fa/image%g.tif',iFa) ;
        newName = sprintf('./train/image%g.tif',index) ;
        copyfile(originalName,newName)
        
        solution(index,1) = index ;
        solution(index,2) = 0 ;
        
        iFa = iFa + 1 ;
        index = index + 1 ;
        
    end
    
    if (iTarget <= maxTarget) && rVal
        originalName = sprintf('./target/image%g.tif',iTarget) ;
        newName = sprintf('./train/image%g.tif',index) ;
        copyfile(originalName,newName)
        
        solution(index,1) = index ;
        solution(index,2) = 1 ;
        
        iTarget = iTarget + 1 ;
        index = index + 1 ;
    end
end

csvwrite('train_solution.csv',solution)

% % Validate
% d = dir('./train') ;
% nValues = length(d) - 2 ;
% k = 1 ;
% while k <= nValues
%     im = imread(['./train/image' num2str(k) '.tif']) ;
%     image(im)
%     fprintf('%g\n',solution(k,2))
%     pause
%     k = k + 1; 
% end