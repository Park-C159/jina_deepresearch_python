import re
import urllib.parse
import aiohttp
import asyncio
from utils.get_log import get_logger

import re
import urllib.parse


def normalize_url(
        url_string: str,
        debug: bool = False,
        options: dict = None
) -> str | None:
    """
    Normalize a URL string according to similar rules as the TS version.
    Returns normalized URL or None if invalid.
    """
    if options is None:
        options = {
            'removeAnchors': True,
            'removeSessionIDs': True,
            'removeUTMParams': True,
            'removeTrackingParams': True,
            'removeXAnalytics': True
        }

    def log_debug(msg, obj):
        print('[DEBUG]', msg, obj)

    def log_warning(msg):
        print('[WARN]', msg)

    try:
        url_string = re.sub(r'\s+', '', url_string).strip()
        if not url_string:
            raise ValueError("Empty URL")
        if url_string.startswith('https://google.com/') or url_string.startswith(
                'https://www.google.com') or url_string.startswith('https://baidu.com/s?'):
            raise ValueError('Google/baidu search link')
        if 'example.com' in url_string:
            raise ValueError('Example URL')

        # x.com / twitter.com /analytics
        if options.get('removeXAnalytics', True):
            x_com_pattern = re.compile(
                r'^(https?://(?:www\.)?(x\.com|twitter\.com)/([^/]+)/status/(\d+))/analytics(/)?(\?.*)?(#.*)?$',
                re.IGNORECASE
            )
            match = x_com_pattern.match(url_string)
            if match:
                clean_url = match.group(1)
                if match.group(6): clean_url += match.group(6)
                if match.group(7): clean_url += match.group(7)
                url_string = clean_url

        # Parse URL
        try:
            url = urllib.parse.urlparse(url_string)
        except Exception:
            raise ValueError('URL parse error')

        if url.scheme not in ('http', 'https'):
            raise ValueError('Unsupported protocol')

        hostname = url.hostname.lower() if url.hostname else ''
        if hostname.startswith('www.'):
            hostname = hostname[4:]

        port = url.port
        # Remove default port
        netloc = hostname
        if port and not ((url.scheme == 'http' and port == 80) or (url.scheme == 'https' and port == 443)):
            netloc += f':{port}'

        # Path normalization
        def decode_seg(seg):
            try:
                return urllib.parse.unquote(seg)
            except Exception as e:
                if debug:
                    log_debug(f'Failed to decode path segment: {seg}', {'error': str(e)})
                return seg

        path = '/'.join([decode_seg(s) for s in url.path.split('/')])
        path = re.sub(r'/+', '/', path)
        path = re.sub(r'/+$', '', path)
        path = path if path else '/'

        # Query param normalization
        search_params = urllib.parse.parse_qsl(url.query, keep_blank_values=True)

        def param_filter(key):
            if key == '':
                return False
            # Session IDs
            if options.get('removeSessionIDs', True) and re.match(
                    r'^(s|session|sid|sessionid|phpsessid|jsessionid|aspsessionid|asp\.net_sessionid)$', key, re.I):
                return False
            # UTM params
            if options.get('removeUTMParams', True) and re.match(r'^utm_', key, re.I):
                return False
            # tracking params
            if options.get('removeTrackingParams', True) and re.match(
                    r'^(ref|referrer|fbclid|gclid|cid|mcid|source|medium|campaign|term|content|sc_rid|mc_[a-z]+)$', key,
                    re.I):
                return False
            return True

        sorted_params = []
        for key, value in search_params:
            if not param_filter(key):
                continue
            try:
                decoded_value = urllib.parse.unquote(value)
                if urllib.parse.quote(decoded_value) == value:
                    value = decoded_value
            except Exception as e:
                if debug:
                    log_debug(f'Failed to decode query param {key}={value}', {'error': str(e)})
            sorted_params.append((key, value))

        sorted_params = sorted(sorted_params, key=lambda kv: kv[0])
        query = urllib.parse.urlencode(sorted_params)

        # Fragment normalization
        fragment = url.fragment
        if options.get('removeAnchors', True):
            fragment = ''
        elif fragment in ('', '/', 'top'):
            fragment = ''
        else:
            try:
                decoded_frag = urllib.parse.unquote(fragment)
                if urllib.parse.quote(decoded_frag) == fragment:
                    fragment = decoded_frag
            except Exception as e:
                if debug:
                    log_debug(f'Failed to decode fragment: #{fragment}', {'error': str(e)})

        # Build final URL
        final_url = urllib.parse.urlunparse((
            url.scheme,
            netloc,
            path,
            '',  # params
            query,
            fragment
        ))

        # Remove trailing slash except for home "/"
        if path != '/' and path.endswith('/'):
            path = path[:-1]
            final_url = urllib.parse.urlunparse((
                url.scheme,
                netloc,
                path,
                '',  # params
                query,
                fragment
            ))

        # Final decode check
        try:
            decoded_url = urllib.parse.unquote(final_url)
            if urllib.parse.quote(decoded_url) == final_url:
                final_url = decoded_url
        except Exception as e:
            if debug:
                log_debug('Failed to decode final URL', {'error': str(e)})

        return final_url
    except Exception as e:
        log_warning(f'Invalid URL "{url_string}": {e}')
        return None


async def get_last_modified(url: str) -> str | None:
    """
    Fetch last modified date for a URL via external API.
    """
    api_url = f'https://api-beta-datetime.jina.ai?url={urllib.parse.quote(url)}'
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, timeout=10) as resp:
                data = await resp.json()
                if data.get('bestGuess') and data.get('confidence', 0) >= 70:
                    return data['bestGuess']
        return None
    except Exception as e:
        print('[ERROR] Failed to fetch last modified date', e)
        return None


def smart_merge_strings(str1: str, str2: str) -> str:
    # If either string is empty, return the other
    if not str1:
        return str2
    if not str2:
        return str1

    # Check if one string is entirely contained within the other
    if str2 in str1:
        return str1
    if str1 in str2:
        return str2

    # Find the maximum possible overlap length
    max_overlap = min(len(str1), len(str2))
    best_overlap_length = 0

    # Check for overlaps starting from the largest possible
    for overlap_length in range(max_overlap, 0, -1):
        end_of_str1 = str1[-overlap_length:]
        start_of_str2 = str2[:overlap_length]
        if end_of_str1 == start_of_str2:
            best_overlap_length = overlap_length
            break

    # If found overlap, merge without repeating that part
    if best_overlap_length > 0:
        return str1 + str2[best_overlap_length:]
    else:
        return str1 + str2


def add_to_all_urls(r, all_urls, weight_delta=1):
    """
    r: SearchSnippet对象（dict或自定义类）
    all_urls: dict，key=标准化后的url，value=SearchSnippet对象
    weight_delta: 权重增量
    返回值: 如果新加入URL返回1，否则返回0
    """
    n_url = normalize_url(r['url'])
    if not n_url:
        return 0
    if n_url not in all_urls:
        all_urls[n_url] = r.copy()  # 确保新的对象
        all_urls[n_url]['weight'] = weight_delta
        return 1
    else:
        all_urls[n_url]['weight'] += weight_delta
        cur_desc = all_urls[n_url]['description']
        # 合并旧描述和新描述
        all_urls[n_url]['description'] = smart_merge_strings(cur_desc, r['description'])
        return 0
