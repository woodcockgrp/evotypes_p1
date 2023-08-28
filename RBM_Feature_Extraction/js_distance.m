function d = js_distance(XI,XJ)

    % function to calculate the Jensen-Shannon distance

    m=size(XJ,1); % number of samples of p
    p=size(XI,2); % dimension of samples

    assert(p == size(XJ,2)); % equal dimensions
    assert(size(XI,1) == 1); % pdist requires XI to be a single sample

    d=zeros(m,1); % initialize output array

    for i=1:m
        for j=1:p
            m=(XJ(i,j) + XI(1,j)) / 2;
            if m ~= 0  % if m == 0, then xi == xj == 0
                d(i,1) = d(i,1) + (XI(1,j) * log(XI(1,j) / m)) + (XJ(i,j) * log(XJ(i,j) / m));
            end
        end
    end

    d=d/2;
    %d=sqrt(d);
end

