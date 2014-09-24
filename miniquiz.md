1. Your company is testing out a new version of the site. They would like to see if the new version results in more friend requests sent. They decided to A/B test it by randomly giving 5% of hits the new site. The results are in these SQL tables. The experiment was run from 2014-08-21 to 2014-08-28 so you should only include friend request data over that time frame.

    ```
    landing_page_test
        userid
        group

    logins
        userid
        date

    friend_requests
        userid
        recipient
        date
    ```

    The `landing_page_test` has a group for every useid, either `new_page` or `old_page`.

    Write a SQL query (or queries) to get the data to fill in the following table.

    |    group | number of logins | number of friend requests |
    | -------- | ---------------- | ------------------------- |
    | new page |                  |                           |
    | old page |                  |                           |



2. Now that you've collected the data, let's say here's the results you pulled from the SQL tables:

    |    group | number of logins | number of friend requests |
    | -------- | ---------------- | ------------------------- |
    | new page |            51982 |                       680 |
    | old page |          1039410 |                     12801 |


    Are you confident that the new landing page is better? Show your work with both a frequentist and a bayesian approach.

    If not, how would you recommend your team to proceed?

